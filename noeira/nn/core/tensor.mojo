"""Tensor[dt] — the storage cell for the storage-passing design (CPU + GPU).

Owns a CPU `List` AND an optional GPU `DeviceBuffer`; the active one is picked
by the method's `target`. `dt` defaults to `DT` (fp32), so the surface stays
`Tensor`; AMP scratch is `Tensor[DType.bfloat16]`. The KEY idea is unchanged:
leaves/orchestrators pass `ref`/`mut Tensor` (the STORAGE), and each method
builds its typed view INTERNALLY — `TileTensor(self.data, …)` on CPU, or
`self.lt["gpu", layout]()` (a device `LayoutTensor`) on GPU. The only erasure on
the GPU path is the kernel-arg `MutAnyOrigin` (the GPU ABI).
"""

from std.os import getenv
from std.sys import size_of
from max.gpu.host import (
    DeviceContext,
    DeviceBuffer,
    DeviceEvent,
    HostBuffer,
)
from layout import Layout, LayoutTensor, RuntimeLayout

from noeira.nn.constants import DT
from .fill import fill_dev


def _alloc_trace[dt: DType](site: StaticString, n: Int, id: Int):
    """Print every DEVICE allocation when `NOEIRA_ALLOC_TRACE=1`.

    A CUDA-graph capture aborts on the FIRST allocation inside the region, so
    blockers surface one at a time and each fix reveals the next. This finds
    them all at once, eagerly, with no capture involved: run a few steady-state
    steps with the trace on and read what still allocates AFTER the first one.
    Anything printed on step 2+ is a per-step allocation, and a per-step
    allocation is a capture blocker.

    Prints MB the same way MAX's allocator does, so a line here can be matched
    against a `(size: 18.75MB)` in a capture failure by eye.

    ⚠ `id` IS THE POINT — group by it, never by size. A model has many distinct
    buffers of the SAME size (ACT's encoder has ~40 `[2592, 256]` activations),
    so a size appearing 227 times is 227 buffers allocated ONCE just as easily
    as one buffer allocated 227 times, and those are opposite diagnoses. `id`
    is the address of the `TensorImpl` cell, which is stable for a Module
    field, so a REPEATED id is a buffer being reallocated — the thing that
    blocks capture. A repeated SIZE is nothing.

    ⚠ Only covers allocations made through `TensorImpl`. If a steady-state step
    prints NOTHING here and a capture still fails on an allocation, the caller
    is inside MAX — use `MODULAR_DEBUG=stack-trace-on-error` for that one."""
    if getenv("NOEIRA_ALLOC_TRACE", "0") == "0":
        return
    var nbytes = n * size_of[Scalar[dt]]()
    print(
        "[alloc] id=", id, "  ", site, "  n=", n, "  ", String(dt),
        "  ", Float64(nbytes) / (1024.0 * 1024.0), "MB", sep="",
    )


struct TensorImpl[dt: DType = DT](Defaultable & Movable & Deinitable):
    var data: List[Scalar[Self.dt]]
    var dev: Optional[DeviceBuffer[Self.dt]]
    var n: Int  # logical length (tracks the device buffer too)
    # Persistent pinned host staging buffer for H2D/D2H. Lazily (re)allocated
    # by `ensure_host` and REUSED across upload/download so a per-step hot loop
    # doesn't churn (and leak) pinned host buffers. `hcap` is its capacity.
    var hbuf: Optional[HostBuffer[Self.dt]]
    var hcap: Int
    # Monotonic write-version of the VALUES, bumped by the optimizer once per
    # step (on the param-value tensors it updates — see `ParamVersionBump`). AMP
    # leaves read it to invalidate a cached low-precision weight copy: recast iff
    # `val.version` advanced since the last cast (so the bf16 weight is cast ONCE
    # per optimizer step, not once per forward — the Phase-1 economics fix).
    # Inert on activation tensors (never bumped, never read there).
    var version: Int
    var _no_events: Bool
    """Device cannot create events (Metal: "eventCreate is not supported").

    Probed ONCE, on the first `upload_resident`, because it is a device
    capability and not something the target string tells us — `target="gpu"`
    covers both CUDA and Metal here. When set, `upload_resident` falls back to
    the device-wide `ctx.synchronize()` it used to do unconditionally."""
    var _h2d_done: Optional[DeviceEvent]
    """Completion of the LAST H2D copy out of `hbuf` (lazy, `upload_resident`).

    ⚠ The dependency being expressed is "is my pinned staging buffer free to
    overwrite", which is a per-BUFFER question. `upload_resident` used to
    answer it with `ctx.synchronize()` — a full DEVICE drain, which also waits
    for every kernel enqueued after that copy, i.e. the whole previous
    training step. An event recorded right after the copy is already complete
    by the time the next step reaches the fill, so the wait costs nothing and
    the pipeline never drains. Measured on ACT: 8 of the ~28 device
    synchronizations per iteration came from here."""

    def __init__(out self):
        self.data = List[Scalar[Self.dt]]()
        self.dev = None
        self.n = 0
        self.hbuf = None
        self.hcap = 0
        self.version = 0
        self._no_events = False
        self._h2d_done = None

    # ----- unified CPU/GPU allocator -------------------------------------
    @staticmethod
    def make[
        target: StaticString
    ](n: Int, ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified allocator (zero-filled, length `n`) — dispatches to
        `alloc` / `alloc_gpu`. `ctx` is ignored on CPU and required on GPU.
        Lets `[target]`-generic code allocate without a `comptime if` branch
        at every site (the CPU/GPU-path unification the leaves want)."""
        comptime if target == "cpu":
            return Self.alloc(n)
        elif target == "gpu":
            if not ctx:
                raise Error("Tensor.make[target='gpu']: ctx required")
            return Self.alloc_gpu(ctx.value(), n)
        else:
            comptime assert False, "target must be 'cpu' or 'gpu'"

    def ensure[
        target: StaticString
    ](mut self, n: Int, ctx: Optional[DeviceContext] = None) raises:
        """Unified lazy-(re)allocate to >= `n` — dispatches to the target's
        `ensure` / `ensure_gpu`. The `[target]`-generic companion to `make`."""
        comptime if target == "cpu":
            self.ensure(n)
        elif target == "gpu":
            self.ensure_gpu(ctx.value(), n)
        else:
            comptime assert False, "target must be 'cpu' or 'gpu'"

    # ----- CPU -----------------------------------------------------------
    @staticmethod
    def alloc(n: Int) -> Self:
        var t = Self()
        t.data = List[Scalar[Self.dt]](length=n, fill=Scalar[Self.dt](0))
        t.n = n
        return t^

    def release(mut self):
        """Drop the storage (device buffer, host list, pinned staging); the
        next `ensure` / `ensure_gpu` allocates afresh. For buffers a module
        REBUILDS on its next forward / vjp (`Module.release_buffers`).

        ⚠ The new allocation is NOT zero-filled on the device (`ensure_gpu`):
        only release a buffer whose every element is written before it is
        read. A device buffer goes back to MAX's caching pool, so the next
        allocation reuses it without a driver call."""
        self.dev = None
        self.data = List[Scalar[Self.dt]]()
        self.n = 0
        self.hbuf = None
        self.hcap = 0

    def ensure(mut self, n: Int):
        """Lazy-grow the CPU list to >= n, zero-filled."""
        if len(self.data) < n:
            self.data = List[Scalar[Self.dt]](length=n, fill=Scalar[Self.dt](0))
            self.n = n

    # ----- GPU -----------------------------------------------------------
    @staticmethod
    def alloc_gpu(ctx: DeviceContext, n: Int) raises -> Self:
        var t = Self()
        _alloc_trace[Self.dt]("alloc_gpu", n, 0)  # no cell yet: fresh by definition
        t.dev = ctx.enqueue_create_buffer[Self.dt](n)
        # ⚠ NOT `enqueue_fill`: on Metal that is a synchronize (`nn/core/fill.mojo`).
        fill_dev(t.dev.value(), n, Scalar[Self.dt](0), ctx)
        t.n = n
        return t^

    def ensure_gpu(mut self, ctx: DeviceContext, n: Int) raises:
        """Lazy-(re)allocate the device buffer to >= n.

        Allocates only on a GROW, so a buffer whose size the warmup already
        reached is a no-op here and safe inside a capture region. One that is
        still growing on a later step is a capture blocker — `_alloc_trace`
        prints it."""
        if not self.dev or self.n < n:
            _alloc_trace[Self.dt](
                "ensure_gpu", n, Int(Pointer(to=self))
            )
            self.dev = ctx.enqueue_create_buffer[Self.dt](n)
            self.n = n

    @staticmethod
    def view_gpu(
        ctx: DeviceContext,
        ptr: Pointer[Scalar[Self.dt], MutAnyOrigin],
        n: Int,
    ) raises -> Self:
        """BORROWING device view — wrap an EXTERNALLY-owned device buffer (e.g.
        an MCTS planner's search buffer, reached as a `LayoutTensor.ptr`) in a
        non-owning `Tensor` so it can be fed straight to `forward`/`vjp` with NO
        copy. The caller owns the memory and MUST keep it alive for the view's
        lifetime; `owning=False` means this `Tensor` never frees it (it dies as
        a plain handle, the storage outlives it via the owner).

        This is the sanctioned boundary between the storage `Module` surface and
        the raw device-buffer interop of `planners/tree_search` — the adapters
        in `deep_agents/zero/mcts_adapters*` build one of these per net input and
        output, then call `net.forward["gpu", B](TensorRefs(in_view), out_view)`.
        The net's whole forward/vjp then runs on the safe storage surface; only
        this thin wrap touches a raw pointer (irreducible GPU ABI).

        GPU-only: the CPU `data` List owns its storage and cannot alias external
        memory, so CPU adapters copy into an owned `Tensor` instead (no analog —
        like `lt_at`).

        ⚠️ A single non-owning view is fine as a kernel operand, but TWO
        simultaneous non-owning views as operands of ONE kernel miscompile on
        Metal (deterministic prefix-drop — the exclusivity/wildcard-origin class
        of the prior ExternalRef GPU bug). For the adapter's input+output case,
        prefer `copy_from_device`/`copy_to_device` (owned scratch + a small D2D
        copy at the boundary; the external buffer is only ever a copy ENDPOINT,
        never a kernel operand)."""
        var t = Self()
        t.dev = DeviceBuffer[Self.dt](ctx, ptr, n, owning=False)
        t.n = n
        return t^

    def copy_from_device(
        mut self,
        ctx: DeviceContext,
        src: Pointer[Scalar[Self.dt], MutAnyOrigin],
        n: Int,
    ) raises:
        """Device→device copy an EXTERNALLY-owned buffer INTO this Tensor's own
        device buffer (lazily allocated to >= n). The external pointer (e.g. an
        MCTS planner's `LayoutTensor.ptr`) is wrapped in a transient non-owning
        `DeviceBuffer` used ONLY as a copy endpoint — never a kernel operand —
        so the net's forward/vjp then runs entirely on owned storage. This is
        the robust adapter-boundary INPUT bridge (see `view_gpu`'s warning)."""
        self.ensure_gpu(ctx, n)
        var src_buf = DeviceBuffer[Self.dt](ctx, src, n, owning=False)
        ctx.enqueue_copy(self.dev.value(), src_buf)

    def copy_to_device(
        mut self,
        ctx: DeviceContext,
        dst: Pointer[Scalar[Self.dt], MutAnyOrigin],
        n: Int,
    ) raises:
        """Device→device copy this Tensor's own device buffer OUT to an
        EXTERNALLY-owned buffer (the planner side). Mirror of
        `copy_from_device`; the adapter-boundary OUTPUT bridge."""
        var dst_buf = DeviceBuffer[Self.dt](ctx, dst, n, owning=False)
        ctx.enqueue_copy(dst_buf, self.dev.value())

    def ensure_host(mut self, ctx: DeviceContext, n: Int) raises:
        """Lazy-(re)allocate the pinned host staging buffer to >= n. Reused
        across upload/download so the hot loop doesn't churn pinned buffers."""
        if not self.hbuf or self.hcap < n:
            self.hbuf = ctx.enqueue_create_host_buffer[Self.dt](n)
            self.hcap = n

    def lt[
        target: StaticString, layout: Layout
    ](mut self) -> LayoutTensor[Self.dt, layout, MutAnyOrigin]:
        """Typed device view at `layout`. Origin-linking ctor (no
        `.unsafe_ptr()`); `MutAnyOrigin` is the GPU kernel-ABI boundary."""
        comptime if target == "cpu":
            return LayoutTensor[Self.dt, layout, MutAnyOrigin](self.data)
        elif target == "gpu":
            return LayoutTensor[Self.dt, layout, MutAnyOrigin](self.dev.value())
        else:
            comptime assert False, "target must be 'cpu' or 'gpu'"

    def lt_dyn[
        target: StaticString, layout: Layout
    ](mut self, rl: RuntimeLayout[layout]) -> LayoutTensor[
        Self.dt, layout, MutAnyOrigin
    ]:
        """The same view, shaped at RUN TIME — `layout` carries UNKNOWN
        extents and `rl` carries the real ones.

        This is `lt`'s counterpart for physics3d's dynamic leg (assessment
        §12.4): one kernel body serves a comptime `Layout.row_major(BATCH,
        NV*NV)` from the GPU path and a `Layout.row_major[2]()` + this from
        the CPU path, because `LM: Layout` accepts both. Same origin-linking
        ctor as `lt` — no `.unsafe_ptr()`, so the borrow is not severed."""
        comptime if target == "cpu":
            return LayoutTensor[Self.dt, layout, MutAnyOrigin](self.data, rl)
        elif target == "gpu":
            return LayoutTensor[Self.dt, layout, MutAnyOrigin](
                self.dev.value(), rl
            )
        else:
            comptime assert False, "target must be 'cpu' or 'gpu'"

    def lt_at[
        target: StaticString, layout: Layout
    ](mut self, offset: Int) raises -> LayoutTensor[
        Self.dt, layout, MutAnyOrigin
    ]:
        """Typed GPU view at element `offset` into the device buffer — a
        stacked-ensemble / per-step sub-view — WITHOUT `.unsafe_ptr()`. The
        sanctioned replacement for
        `LayoutTensor[..MutAnyOrigin](buf.dev.value().unsafe_ptr() + offset)`:
        a memory-sharing `create_sub_buffer` (offset + `layout.size()`, in
        elements) feeds the same explicit-`MutAnyOrigin` DeviceBuffer ctor `lt`
        uses, so the returned static `layout` matches the kernel ABI exactly
        (callers/kernels identical to `lt`, just offset).

        GPU-only: the per-member offset views that needed raw pointers are all on
        the GPU kernel path; CPU branches index the owning `self.data` List
        directly (no ABI erasure, so no helper needed). The parent buffer owns
        the storage — the sub-buffer handle dying after this returns is safe
        (the `MutAnyOrigin` cast erases its origin; the memory outlives via the
        parent)."""
        comptime assert target == "gpu", (
            "lt_at is GPU-only (offset sub-view); CPU indexes self.data directly"
        )
        comptime sz = layout.size()
        var sub = self.dev.value().create_sub_buffer[Self.dt](offset, sz)
        return LayoutTensor[Self.dt, layout, MutAnyOrigin](sub)

    def upload(mut self, ctx: DeviceContext) raises:
        """CPU `data` → device buffer (via the persistent host staging buffer).
        The device buffer is (re)allocated to `self.n` (original semantics —
        callers may grow `self.n` before calling); the pinned host buffer is
        REUSED across calls instead of allocated fresh each time.

        ⚠ This reallocates on EVERY call — by design (it is the resize path) —
        so it is a capture blocker AND a replay hazard (the device pointer
        changes, and a captured graph holds the old one). Under capture use
        `upload_resident`."""
        _alloc_trace[Self.dt](
            "upload (REALLOCATES EVERY CALL)", self.n,
            Int(Pointer(to=self)),
        )
        self.dev = ctx.enqueue_create_buffer[Self.dt](self.n)
        self.ensure_host(ctx, self.n)
        var hb = self.hbuf.value()
        ctx.synchronize()
        for i in range(self.n):
            hb[i] = self.data[i]
        ctx.enqueue_copy(self.dev.value(), hb)
        ctx.synchronize()

    def upload_resident(mut self, ctx: DeviceContext) raises:
        """CPU `data[:n]` → the EXISTING device buffer WITHOUT reallocating it
        (the buffer must already exist and be sized >= `n`; lazily allocates on
        the first call only). Unlike `upload` — which recreates `self.dev` every
        call (changing the pointer) — this reuses the buffer, so a CUDA-graph
        that captured this buffer stays valid across replays: only the CONTENTS
        change. Use as the EAGER per-step input refresh under graph capture."""
        self.ensure_gpu(ctx, self.n)
        self.ensure_host(ctx, self.n)
        var hb = self.hbuf.value()
        # ⚠ Wait for OUR last copy out of `hbuf`, not for the device. See
        # `_h2d_done`. First call: nothing is in flight, so nothing to wait on.
        if self._h2d_done:
            self._h2d_done.value().synchronize()
        elif self._no_events:
            ctx.synchronize()
        for i in range(self.n):
            hb[i] = self.data[i]
        ctx.enqueue_copy(self.dev.value(), hb)
        if not self._h2d_done and not self._no_events:
            try:
                self._h2d_done = ctx.create_event()
            except:
                # Metal. Fall back to the device drain from here on.
                self._no_events = True
        if self._h2d_done:
            try:
                ctx.stream().record_event(self._h2d_done.value())
            except:
                # ⚠ THE FALLBACK CANNOT BE KEYED ON `create_event` ALONE. MAX
                # 26.6 on Metal HANDS OUT an event and then refuses the stream
                # ("Metal stream not implemented"), so the earlier `except`
                # never fires and this one does. The event is dropped rather
                # than kept: an event that was never recorded would be waited
                # on at the top of the next call.
                self._h2d_done = None
                self._no_events = True

    def download_enqueue(mut self, ctx: DeviceContext) raises:
        """Enqueue the D2H copy into the persistent host buffer WITHOUT
        synchronizing. Pair with a later `ctx.synchronize()` + per-tensor
        `download_finalize()` to batch several D2H copies behind ONE sync."""
        self.ensure_host(ctx, self.n)
        ctx.enqueue_copy(self.hbuf.value(), self.dev.value())

    def download_finalize(mut self):
        """Copy the staged host buffer into CPU `data` (call AFTER the sync)."""
        var hb = self.hbuf.value()
        if len(self.data) < self.n:
            self.data = List[Scalar[Self.dt]](
                length=self.n, fill=Scalar[Self.dt](0)
            )
        for i in range(self.n):
            self.data[i] = hb[i]

    def download(mut self, ctx: DeviceContext) raises:
        """Device buffer → CPU `data` (via the persistent host staging buffer).
        The pinned host buffer is reused across calls — the previous design
        allocated a fresh one every call, churning/leaking it in a hot loop."""
        self.download_enqueue(ctx)
        ctx.synchronize()
        self.download_finalize()


# The fp32 surface type is a concrete alias, so bare `Tensor` is never an
# unbound-parameter ambiguity; AMP scratch uses `TensorImpl[DType.bfloat16]`.
comptime Tensor = TensorImpl[DT]
