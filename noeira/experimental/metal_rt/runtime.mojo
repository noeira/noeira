"""A small Metal kernel launcher in pure Mojo.

    var rt = MetalRT(policy=COMMIT_ADAPTIVE, batch_n=16)
    var x = rt.alloc[DType.float32](1024)         # shared storage
    x.host_ptr[DType.float32]()[unsafe_offset=0] = 1.0
    rt.launch[my_kernel](x.dev[DType.float32](), Int64(1024), grid=4, block=256)
    rt.sync()                                      # commit + wait
    print(x.host_ptr[DType.float32]()[unsafe_offset=0])

Built from Apple's public Metal headers only (MTLDevice.h, MTLCommandQueue.h,
MTLCommandBuffer.h, MTLComputeCommandEncoder.h, MTLEvent.h,
MTLResidencySet.h) plus the Mojo stdlib. The kernel is compiled to a metallib
by `_compile_code` (the same compiler MAX uses), so a kernel launched here and
one launched through `DeviceContext.enqueue_function` run the same code.

## Kernel ABI (measured, see `test_abi.mojo`)

Every kernel argument is its own Metal buffer-binding index, in declaration
order, and is passed by value with `setBytes:length:atIndex:`: a device
pointer is its buffer's 8-byte `gpuAddress` (NOT the host `contents` pointer),
an `Int` is 8 bytes, a `Float32` is 4. Thread / block indices come from the
dispatch itself (`dispatchThreadgroups:threadsPerThreadgroup:`).

## Ordering

All launches between two commits go into ONE serial compute encoder. Apple:
"MTLDispatchTypeSerial: command encoder dispatches are executed in dispatched
order", and `memoryBarrierWithScope:` "on a serial encoder is allowed, but
ignored" -- i.e. a serial encoder already orders memory between dispatches.

Between command buffers nothing is ordered for us. Buffers are created
hazard-UNTRACKED (we never bind them, only pass their GPU address), and
consecutive command buffers on one queue DO overlap on the GPU: with no
ordering, 2000 dependent increments lose up to 30% of their updates
(`bench_launch.mojo`, "no ordering"). `order=` picks the fix:

  ORDER_FENCE    (default) one MTLFence; every encoder but the first waits on
                 it at its start and updates it at its end. Free: same cost as
                 the probe's tracked path.
  ORDER_EVENT    shared-event chain (`encodeWaitForEvent:value:` on the
                 predecessor's value). Correct but ~5x slower per command
                 buffer (110 vs 22 us).
  ORDER_TRACKED  hazard-tracked buffers declared per command buffer
                 (`useResources`); Metal orders. Cost grows with live buffers.
  ORDER_NONE     nothing. Loses updates. Measurement only.

## Completion

  DONE_STATUS    (default) retain the last committed command buffer and read
                 its `status`; with ordered command buffers, "last completed"
                 means "all completed". `sync()` spins on it
                 (`spin_sync=False`: `waitUntilCompleted`).
  DONE_EVENT     every command buffer signals a shared event with its
                 sequence number. Costs ~20 us per command buffer on the GPU
                 side (42 vs 22 us at one launch per command buffer).

`sync_event=True` (default, either mode): the command buffer `sync()`
commits signals the event and `sync()` spins on `signaledValue`. One signal
per sync, not per command buffer, and the host sees completion ~80 us sooner
than through `status` (one-launch round trip 108 vs 190 us).

No completion handlers (ObjC blocks) are needed in any mode.

## Commit policies

  COMMIT_EACH      every launch is its own command buffer.
  COMMIT_EVERY_N   commit when `batch_n` launches are pending.
  COMMIT_ADAPTIVE  commit at once if the GPU has nothing in flight
                   (GPU idle => latency matters), else batch until `batch_n`.
                   With `idle_age_us=T > 0`, "at once" becomes "once the
                   oldest pending launch is T us old": a command buffer costs
                   the GPU ~20 us, encoding a launch costs the host ~0.15 us,
                   so committing a lone launch to an idle GPU only pays when
                   the host is slow to send the next one.

Every policy flushes on `sync()` and `flush()`. ADAPTIVE cannot see the future:
launches batched while the GPU was busy stay on the host until the next launch,
`flush()` or `sync()`. A caller about to do long CPU work without syncing
should call `flush()` first (see README, "the tail problem").

## Residency

Metal must know every buffer a command buffer touches through a GPU address.

  RESIDENCY_SET     one MTLResidencySet holding every live buffer, attached
                    to the queue once (macOS 15+). Nothing per command buffer.
  RESIDENCY_PER_CB  `useResources:count:usage:` with every live buffer, once
                    per command buffer.
  RESIDENCY_PER_LAUNCH  the same call before every dispatch (a stand-in for a
                    runtime that declares all live buffers on every launch).
"""

from std.memory import Pointer
from std.sys import size_of
from std.time import perf_counter_ns
from max.gpu import global_idx
from max.gpu.host.compile import _compile_code

from noeira.experimental.metal_rt.objc import ObjC

comptime COMMIT_EACH = 0
comptime COMMIT_EVERY_N = 1
comptime COMMIT_ADAPTIVE = 2

comptime RESIDENCY_SET = 0
comptime RESIDENCY_PER_CB = 1
comptime RESIDENCY_PER_LAUNCH = 2

# How consecutive command buffers are ordered on the GPU.
comptime ORDER_FENCE = 0  # one MTLFence: wait at encoder start, update at end
comptime ORDER_EVENT = 1  # shared event: wait for predecessor's value
comptime ORDER_NONE = 2  # nothing (UNSAFE with untracked buffers; measurement)
comptime ORDER_TRACKED = 3  # hazard-tracked buffers + useResources per cb

# How the host learns that command buffers completed.
comptime DONE_STATUS = 0  # retain the last command buffer, read its `status`
comptime DONE_EVENT = 1  # every command buffer signals a shared event

comptime _SHARED_UNTRACKED = 0 | (1 << 8)  # StorageModeShared | Untracked
comptime _SHARED_TRACKED = 0 | (2 << 8)  # StorageModeShared | Tracked
comptime _USAGE_RW = 1 | 2  # MTLResourceUsageRead | MTLResourceUsageWrite
comptime _STATUS_COMPLETED = 4  # MTLCommandBufferStatusCompleted


def policy_name(p: Int) -> String:
    if p == COMMIT_EACH:
        return "each"
    if p == COMMIT_EVERY_N:
        return "every-N"
    return "adaptive"


def load_pipeline(o: ObjC, device: Int, lib_bytes: String, fname: String) raises -> Int:
    """metallib bytes -> MTLLibrary -> MTLFunction -> MTLComputePipelineState."""
    var pool = o.pool_push()
    var data = o.system.get_function[Int]("dispatch_data_create")(
        Int(lib_bytes.unsafe_ptr()), lib_bytes.byte_length(), 0, 0
    )
    var err = List[Int](length=1, fill=0)
    var errp = Int(err.unsafe_ptr())
    var mlib = o.send2(
        device, o.sel("newLibraryWithData:error:"), data, errp
    )
    o.check_error(err[0], "newLibraryWithData")
    var mfn = o.send1(
        mlib, o.sel("newFunctionWithName:"), o.nsstring(fname)
    )
    if mfn == 0:
        raise Error("function not in metallib: " + fname)
    var pso = o.send2(
        device,
        o.sel("newComputePipelineStateWithFunction:error:"),
        mfn,
        errp,
    )
    o.check_error(err[0], "newComputePipelineStateWithFunction")
    o.release(mfn)
    o.release(mlib)
    o.release(data)
    o.pool_pop(pool)
    return pso


comptime _STAGE_BYTES = 4 << 20


def _copy_words(
    dst: Pointer[UInt32, MutAnyOrigin], src: Pointer[UInt32, MutAnyOrigin], n: Int64
):
    var i = Int(global_idx.x)
    if i < Int(n):
        dst[unsafe_offset=i] = src[unsafe_offset=i]


def _memcpy(dst: Int, src: Int, nbytes: Int):
    var d = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=dst)
    var s = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=src)
    var i = 0
    if (dst | src | nbytes) & 7 == 0:
        var d8 = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=dst)
        var s8 = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=src)
        for k in range(nbytes // 8):
            d8[unsafe_offset=k] = s8[unsafe_offset=k]
        return
    while i < nbytes:
        d[unsafe_offset=i] = s[unsafe_offset=i]
        i += 1


trait Launcher(Movable & Deinitable):
    """What `MetalRT` and `Metal4RT` both offer, so a workload can be written
    once and run on either."""

    def launch[
        func_type: TrivialRegisterPassable, //, func: func_type, *Ts: AnyType
    ](mut self, *args: *Ts, grid: Int, block: Int) raises:
        ...

    def alloc[dt: DType](mut self, count: Int) raises -> Buffer:
        ...

    def flush(mut self):
        ...

    def sync(mut self):
        ...

    def commits(self) -> Int:
        ...

    def idle_commits(self) -> Int:
        ...


@fieldwise_init
struct Buffer(ImplicitlyCopyable, Movable):
    """A shared-storage MTLBuffer. `host` is the CPU address (`contents`),
    `gpu` the GPU virtual address (`gpuAddress`); they differ."""

    var obj: Int
    var host: Int
    var gpu: Int
    var nbytes: Int

    def host_ptr[dt: DType](self) -> Pointer[Scalar[dt], MutAnyOrigin]:
        return Pointer[Scalar[dt], MutAnyOrigin](unsafe_from_address=self.host)

    def dev[dt: DType](self) -> Pointer[Scalar[dt], MutAnyOrigin]:
        """The kernel-argument form: the GPU address typed as a pointer. Never
        dereference it on the host."""
        return Pointer[Scalar[dt], MutAnyOrigin](unsafe_from_address=self.gpu)


struct _Sel(Movable):
    """Every selector a launch touches, registered once."""

    var command_buffer_unretained: Int
    var compute_encoder: Int
    var set_pso: Int
    var set_bytes: Int
    var dispatch: Int
    var end_encoding: Int
    var commit: Int
    var wait_event: Int
    var signal_event: Int
    var signaled_value: Int
    var wait_fence: Int
    var update_fence: Int
    var use_resources: Int
    var add_allocation: Int
    var status: Int
    var retain: Int
    var release: Int
    var wait_until_completed: Int

    def __init__(out self, o: ObjC):
        self.command_buffer_unretained = o.sel(
            "commandBufferWithUnretainedReferences"
        )
        self.compute_encoder = o.sel("computeCommandEncoder")
        self.set_pso = o.sel("setComputePipelineState:")
        self.set_bytes = o.sel("setBytes:length:atIndex:")
        self.dispatch = o.sel("dispatchThreadgroups:threadsPerThreadgroup:")
        self.end_encoding = o.sel("endEncoding")
        self.commit = o.sel("commit")
        self.wait_event = o.sel("encodeWaitForEvent:value:")
        self.signal_event = o.sel("encodeSignalEvent:value:")
        self.signaled_value = o.sel("signaledValue")
        self.wait_fence = o.sel("waitForFence:")
        self.update_fence = o.sel("updateFence:")
        self.use_resources = o.sel("useResources:count:usage:")
        self.add_allocation = o.sel("addAllocation:")
        self.status = o.sel("status")
        self.retain = o.sel("retain")
        self.release = o.sel("release")
        self.wait_until_completed = o.sel("waitUntilCompleted")


struct MetalRT(Launcher):
    var o: ObjC
    var s: _Sel
    var device: Int
    var queue: Int
    var event: Int
    var fence: Int
    var rset: Int
    var policy: Int
    var batch_n: Int
    var residency: Int
    var order: Int
    var done: Int
    var spin_sync: Bool
    var sync_event: Bool
    var last_signaled: Int  # highest event value any command buffer signals
    var idle_age_ns: Int
    var first_pending_ns: Int  # when the open command buffer got its first launch
    var cb: Int
    var enc: Int
    var pool: Int
    var last_cb: Int  # retained: the most recently committed command buffer
    var pending: Int  # launches encoded into the open command buffer
    var committed: Int  # command buffers committed
    var cur_pso: Int
    var rset_dirty: Bool
    var live: List[Int]  # every live MTLBuffer (for the per-cb modes)
    var live_gpu: List[Int]
    var live_host: List[Int]
    var live_bytes: List[Int]
    var stage: Buffer  # upload staging ring (allocated on first upload)
    var stage_off: Int
    var _k_keys: List[Int]
    var _k_pso: List[Int]
    var _dims: List[Int]  # grid xyz, block xyz (MTLSize by reference)
    # stats
    var n_launches: Int
    var n_commits: Int
    var n_idle_commits: Int  # ADAPTIVE commits taken because the GPU was idle

    def __init__(
        out self,
        policy: Int = COMMIT_ADAPTIVE,
        batch_n: Int = 16,
        residency: Int = RESIDENCY_SET,
        order: Int = ORDER_FENCE,
        done: Int = DONE_STATUS,
        spin_sync: Bool = True,
        idle_age_us: Int = 0,
        sync_event: Bool = True,
    ) raises:
        self.o = ObjC()
        self.s = _Sel(self.o)
        self.device = self.o.metal.get_function[Int](
            "MTLCreateSystemDefaultDevice"
        )()
        if self.device == 0:
            raise Error("no Metal device")
        self.queue = self.o.send0(self.device, self.o.sel("newCommandQueue"))
        self.event = self.o.send0(self.device, self.o.sel("newSharedEvent"))
        self.fence = self.o.send0(self.device, self.o.sel("newFence"))
        self.policy = policy
        self.batch_n = batch_n
        self.residency = residency
        if order == ORDER_TRACKED and residency == RESIDENCY_SET:
            # Tracking needs every resource declared to the encoder.
            self.residency = RESIDENCY_PER_CB
        self.order = order
        self.done = done
        self.spin_sync = spin_sync
        self.sync_event = sync_event
        self.last_signaled = 0
        self.idle_age_ns = idle_age_us * 1000
        self.first_pending_ns = 0
        self.cb = 0
        self.enc = 0
        self.pool = 0
        self.last_cb = 0
        self.pending = 0
        self.committed = 0
        self.cur_pso = 0
        self.rset_dirty = False
        self.live = List[Int]()
        self.live_gpu = List[Int]()
        self.live_host = List[Int]()
        self.live_bytes = List[Int]()
        self.stage = Buffer(0, 0, 0, 0)
        self.stage_off = 0
        self._k_keys = List[Int]()
        self._k_pso = List[Int]()
        self._dims = List[Int](length=6, fill=1)
        self.n_launches = 0
        self.n_commits = 0
        self.n_idle_commits = 0
        self.rset = 0
        if self.residency == RESIDENCY_SET:
            var desc = self.o.send0(
                self.o.send0(
                    self.o.cls("MTLResidencySetDescriptor"), self.o.sel("alloc")
                ),
                self.o.sel("init"),
            )
            _ = self.o.send1(desc, self.o.sel("setInitialCapacity:"), 64)
            var err = List[Int](length=1, fill=0)
            self.rset = self.o.send2(
                self.device,
                self.o.sel("newResidencySetWithDescriptor:error:"),
                desc,
                Int(err.unsafe_ptr()),
            )
            self.o.check_error(err[0], "newResidencySetWithDescriptor")
            self.o.release(desc)
            _ = self.o.send1(self.queue, self.o.sel("addResidencySet:"), self.rset)

    def __deinit__(deinit self):
        """Wait for the GPU, then release every Metal object we own."""
        self._commit()
        if self.committed > 0:
            while not self.gpu_idle():
                pass
        for i in range(len(self.live)):
            self.o.release(self.live[i])
        for i in range(len(self._k_pso)):
            self.o.release(self._k_pso[i])
        if self.last_cb != 0:
            self.o.release(self.last_cb)
        self.o.release(self.rset)
        self.o.release(self.fence)
        self.o.release(self.event)
        self.o.release(self.queue)

    def commits(self) -> Int:
        return self.n_commits

    def idle_commits(self) -> Int:
        return self.n_idle_commits

    def name(self) -> String:
        return self.o.utf8(self.o.send0(self.device, self.o.sel("name")))

    # ---------------------------------------------------------------- buffers

    def alloc[dt: DType](mut self, count: Int) raises -> Buffer:
        var nbytes = max(count * size_of[Scalar[dt]](), 16)
        var obj = self.o.send2(
            self.device,
            self.o.sel("newBufferWithLength:options:"),
            nbytes,
            _SHARED_TRACKED if self.order == ORDER_TRACKED else _SHARED_UNTRACKED,
        )
        if obj == 0:
            raise Error("newBufferWithLength failed")
        return self._adopt(obj, nbytes)

    def wrap_external(mut self, host_addr: Int, nbytes: Int) raises -> Buffer:
        """A Buffer over memory we do NOT own (an IOSurface's, say), with no
        copy: `newBufferWithBytesNoCopy:length:options:deallocator:` with a
        nil deallocator, shared storage. `host_addr` and `nbytes` must be
        page-aligned (16 KB on Apple silicon). It gets a GPU address and joins
        the residency set like any other buffer; `free` / the destructor
        release the MTLBuffer object only — the memory stays its owner's, who
        must keep it alive while the buffer is in use."""
        if host_addr % 16384 != 0 or nbytes % 16384 != 0 or nbytes == 0:
            raise Error("wrap_external: pointer and length must be 16 KB page multiples")
        var obj = self.o.send4(
            self.device,
            self.o.sel("newBufferWithBytesNoCopy:length:options:deallocator:"),
            host_addr,
            nbytes,
            _SHARED_TRACKED if self.order == ORDER_TRACKED else _SHARED_UNTRACKED,
            0,
        )
        if obj == 0:
            raise Error("newBufferWithBytesNoCopy returned nil")
        var b = self._adopt(obj, nbytes)
        if b.host != host_addr:
            raise Error("wrap_external: contents != the wrapped pointer (Metal copied?)")
        return b

    def _adopt(mut self, obj: Int, nbytes: Int) raises -> Buffer:
        var b = Buffer(
            obj,
            self.o.send0(obj, self.o.sel("contents")),
            self.o.send0(obj, self.o.sel("gpuAddress")),
            nbytes,
        )
        self.live.append(obj)
        self.live_gpu.append(b.gpu)
        self.live_host.append(b.host)
        self.live_bytes.append(nbytes)
        if self.rset != 0:
            # A buffer allocated while a command buffer is OPEN would be used
            # by that command buffer before the residency set's commit took
            # effect: not resident, reads garbage, intermittently (seen on
            # the functional tape, whose Adam moments and scratch appear
            # mid-step). Commit the open command buffer first.
            self._commit()
            _ = self.o.send1(self.rset, self.s.add_allocation, obj)
            self.rset_dirty = True
        return b

    def free(mut self, b: Buffer) raises:
        """Release a buffer. Syncs first: the GPU may still be using it."""
        self.sync()
        for i in range(len(self.live)):
            if self.live[i] == b.obj:
                _ = self.live.pop(i)
                _ = self.live_gpu.pop(i)
                _ = self.live_host.pop(i)
                _ = self.live_bytes.pop(i)
                break
        if self.rset != 0:
            _ = self.o.send1(self.rset, self.o.sel("removeAllocation:"), b.obj)
            self.rset_dirty = True
        self.o.release(b.obj)

    # ---------------------------------------------------------------- kernels

    def _pipeline[func_type: TrivialRegisterPassable, //, func: func_type](mut self) raises -> Int:
        """metallib -> MTLLibrary -> MTLFunction -> pipeline, once per kernel.
        Keyed by the address of the kernel's (static) function-name string."""
        var info = _compile_code[func, emission_kind="object"]()
        var fname = info.function_name
        var key = Int(fname.unsafe_ptr())
        for i in range(len(self._k_keys)):
            if self._k_keys[i] == key:
                return self._k_pso[i]
        var pso = self._load(String(info.asm), String(fname))
        self._k_keys.append(key)
        self._k_pso.append(pso)
        return pso

    def _load(mut self, lib_bytes: String, fname: String) raises -> Int:
        return load_pipeline(self.o, self.device, lib_bytes, fname)

    # ---------------------------------------------------------------- launch

    def launch[
        func_type: TrivialRegisterPassable, //, func: func_type, *Ts: AnyType
    ](mut self, *args: *Ts, grid: Int, block: Int) raises:
        """Encode one dispatch of `grid` threadgroups of `block` threads.
        Argument i is bound at buffer index i by value (see module doc)."""
        self.begin_dispatch[func]()
        comptime for i in range(args.__len__()):
            self.set_arg(i, Int(Pointer(to=args[i])), size_of[Ts[i]]())
        self.dispatch(grid, block)

    def launch2d[
        func_type: TrivialRegisterPassable, //, func: func_type, *Ts: AnyType
    ](
        mut self, *args: *Ts, grid_x: Int, grid_y: Int, block_x: Int,
        block_y: Int = 1,
    ) raises:
        """`launch` over a 2-D grid of threadgroups."""
        self.begin_dispatch[func]()
        comptime for i in range(args.__len__()):
            self.set_arg(i, Int(Pointer(to=args[i])), size_of[Ts[i]]())
        self.dispatch(grid_x, block_x, grid_y, block_y)

    # The same launch in three calls, for a caller that converts its own
    # argument types (`functional/backend.mojo` turns a `DeviceBuffer` into
    # its address): begin, one `set_arg` per argument, dispatch.

    def begin_dispatch[
        func_type: TrivialRegisterPassable, //, func: func_type
    ](mut self) raises:
        var pso = self._pipeline[func]()
        if self.enc == 0:
            self._begin()
            if self.idle_age_ns > 0:
                self.first_pending_ns = Int(perf_counter_ns())
        if pso != self.cur_pso:
            _ = self.o.send1(self.enc, self.s.set_pso, pso)
            self.cur_pso = pso

    @always_inline
    def set_arg(mut self, index: Int, addr: Int, nbytes: Int):
        """Kernel argument `index` = the `nbytes` bytes at host `addr`
        (copied by Metal now)."""
        _ = self.o.send3(self.enc, self.s.set_bytes, addr, nbytes, index)

    def dispatch(
        mut self, grid_x: Int, block_x: Int, grid_y: Int = 1, block_y: Int = 1
    ):
        if self.residency == RESIDENCY_PER_LAUNCH:
            self._use_all()
        self._dims[0] = grid_x
        self._dims[1] = grid_y
        self._dims[3] = block_x
        self._dims[4] = block_y
        var d = Int(self._dims.unsafe_ptr())
        _ = self.o.send2(self.enc, self.s.dispatch, d, d + 24)
        self.pending += 1
        self.n_launches += 1
        self._maybe_commit()

    # ---------------------------------------------------------------- copies

    def host_of(self, gpu_addr: Int) raises -> Int:
        """The host address of a GPU address inside one of our buffers."""
        for i in range(len(self.live)):
            var g = self.live_gpu[i]
            if gpu_addr >= g and gpu_addr < g + self.live_bytes[i]:
                return self.live_host[i] + (gpu_addr - g)
        raise Error("MetalRT.host_of: not an address of a MetalRT buffer")

    def upload(mut self, dst_gpu: Int, src_host: Int, nbytes: Int) raises:
        """Host -> device, ORDERED like a launch: the bytes go into a staging
        ring now and a copy kernel moves them when the GPU gets there, so
        nothing the GPU has yet to read is overwritten under it. A wrap of
        the ring syncs first. `nbytes` is a multiple of 4."""
        if nbytes <= 0:
            return
        if nbytes > _STAGE_BYTES:
            self.sync()
            var dh = self.host_of(dst_gpu)
            _memcpy(dh, src_host, nbytes)
            return
        if self.stage.obj == 0:
            self.stage = self.alloc[DType.uint8](_STAGE_BYTES)
        if self.stage_off + nbytes > _STAGE_BYTES:
            self.sync()
            self.stage_off = 0
        _memcpy(self.stage.host + self.stage_off, src_host, nbytes)
        var nw = Int64(nbytes // 4)
        var src = Pointer[UInt32, MutAnyOrigin](
            unsafe_from_address=self.stage.gpu + self.stage_off
        )
        var dst = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=dst_gpu)
        self.launch[_copy_words](dst, src, nw, grid=(Int(nw) + 255) // 256, block=256)
        self.stage_off += (nbytes + 15) & ~15

    def download(mut self, dst_host: Int, src_gpu: Int, nbytes: Int) raises:
        """Device -> host: sync, then copy out of shared memory."""
        self.sync()
        _memcpy(dst_host, self.host_of(src_gpu), nbytes)

    def gpu_idle(self) -> Bool:
        """True when every committed command buffer has completed.

        Reads the shared event when the last command buffer signalled it, else
        the last command buffer's `status`. `status` turns Completed ~80 us
        after the event does: reading it right after a `sync()` that waited on
        the event says "busy" about a GPU that is idle."""
        if self.done == DONE_EVENT or self.last_signaled == self.committed:
            return (
                self.o.send0(self.event, self.s.signaled_value) >= self.committed
            )
        if self.last_cb == 0:
            return True
        return self.o.send0(self.last_cb, self.s.status) >= _STATUS_COMPLETED

    def _maybe_commit(mut self):
        if self.policy == COMMIT_EACH:
            self._commit()
        elif self.policy == COMMIT_EVERY_N:
            if self.pending >= self.batch_n:
                self._commit()
        else:
            if self.pending >= self.batch_n:
                self._commit()
            elif (
                self.idle_age_ns == 0
                or Int(perf_counter_ns()) - self.first_pending_ns
                >= self.idle_age_ns
            ) and self.gpu_idle():
                self.n_idle_commits += 1
                self._commit()

    def _use_all(mut self):
        _ = self.o.send3(
            self.enc,
            self.s.use_resources,
            Int(self.live.unsafe_ptr()),
            len(self.live),
            _USAGE_RW,
        )

    def _begin(mut self):
        if self.rset_dirty:
            _ = self.o.send0(self.rset, self.s.commit)  # MTLResidencySet commit
            self.rset_dirty = False
        # The command buffer and its encoder come back autoreleased. They are
        # RETAINED here and released at commit, with a pool around each of
        # the two calls only: a pool spanning an open command buffer breaks
        # as soon as two launchers interleave (popping one pool pops every
        # pool pushed after it, releasing the other launcher's open encoder:
        # "Command encoder released without endEncoding").
        var pool = self.o.pool_push()
        self.cb = self.o.send0(
            self.o.send0(self.queue, self.s.command_buffer_unretained),
            self.s.retain,
        )
        if self.order == ORDER_EVENT and self.committed > 0:
            _ = self.o.send2(self.cb, self.s.wait_event, self.event, self.committed)
        self.enc = self.o.send0(
            self.o.send0(self.cb, self.s.compute_encoder), self.s.retain
        )
        if self.order == ORDER_FENCE and self.committed > 0:
            _ = self.o.send1(self.enc, self.s.wait_fence, self.fence)
        self.o.pool_pop(pool)
        self.cur_pso = 0
        if self.residency == RESIDENCY_PER_CB:
            self._use_all()

    def _commit(mut self, signal: Bool = False):
        if self.cb == 0:
            return
        var pool = self.o.pool_push()
        if self.order == ORDER_FENCE:
            _ = self.o.send1(self.enc, self.s.update_fence, self.fence)
        _ = self.o.send0(self.enc, self.s.end_encoding)
        self.committed += 1
        if signal or self.done == DONE_EVENT or self.order == ORDER_EVENT:
            _ = self.o.send2(self.cb, self.s.signal_event, self.event, self.committed)
            self.last_signaled = self.committed
        _ = self.o.send0(self.cb, self.s.commit)
        _ = self.o.send0(self.enc, self.s.release)
        if self.done == DONE_STATUS:
            # our retain moves to `last_cb`, kept for its `status`
            if self.last_cb != 0:
                _ = self.o.send0(self.last_cb, self.s.release)
            self.last_cb = self.cb
        else:
            _ = self.o.send0(self.cb, self.s.release)
        self.o.pool_pop(pool)
        self.cb = 0
        self.enc = 0
        self.pool = 0
        self.pending = 0
        self.n_commits += 1

    def flush(mut self):
        """Commit whatever is encoded; do not wait. The command buffer signals
        the event (~20 us of GPU time) so that a later `sync()` with nothing
        left to commit can still wait on the event rather than on `status`."""
        self._commit(signal=self.sync_event)

    def sync(mut self):
        """Commit, then wait until the GPU has finished everything.

        With `sync_event` (default) the command buffer committed here signals
        the shared event and the host spins on `signaledValue`: the host sees
        that ~80 us sooner than the command buffer's `status` turning
        Completed (bench_launch.mojo, 3b). If nothing was open, the last
        command buffer may not have signalled; fall back to its `status`."""
        self._commit(signal=self.sync_event)
        if self.last_signaled == self.committed:
            while (
                self.o.send0(self.event, self.s.signaled_value) < self.committed
            ):
                pass
        elif self.spin_sync:
            while not self.gpu_idle():
                pass
        elif self.last_cb != 0:
            _ = self.o.send0(self.last_cb, self.s.wait_until_completed)
