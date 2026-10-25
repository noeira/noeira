"""A small Metal kernel launcher in pure Mojo.

    var rt = MetalRT(policy=COMMIT_ADAPTIVE, batch_n=16)
    var x = rt.alloc[DType.float32](1024)         # shared storage
    x.host_ptr[DType.float32]()[unsafe_offset=0] = 1.0
    rt.launch[my_kernel](x.dev[DType.float32](), 1024, grid=4, block=256)
    rt.sync()                                      # commit + wait
    print(x.host_ptr[DType.float32]()[unsafe_offset=0])

Built from Apple's public Metal headers only (MTLDevice.h, MTLCommandQueue.h,
MTLCommandBuffer.h, MTLComputeCommandEncoder.h, MTLEvent.h,
MTLResidencySet.h) plus the Mojo stdlib. The kernel is compiled to a metallib
by `_compile_code` (the same compiler MAX uses), so a kernel launched here and
one launched through `DeviceContext.enqueue_function` run the same code.

## Kernel ABI (measured, see `test_metal_rt_abi.mojo`)

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
Buffers are created hazard-UNTRACKED (we never bind them, only pass their GPU
address), so Metal knows nothing about dependencies between command buffers:
each command buffer waits on the shared event for its predecessor's value
(`encodeWaitForEvent:value:`) before its encoder starts, and signals its own
sequence number when it ends. The same event gives the host the number of
command buffers in flight (`committed - signaledValue`) without completion
handlers.

## Commit policies

  COMMIT_EACH      every launch is its own command buffer.
  COMMIT_EVERY_N   commit when `batch_n` launches are pending.
  COMMIT_ADAPTIVE  commit at once if the GPU has nothing in flight
                   (GPU idle => latency matters), else batch until `batch_n`.

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
from max.gpu.host.compile import _compile_code

from noeira.experimental.metal_rt.objc import ObjC

comptime COMMIT_EACH = 0
comptime COMMIT_EVERY_N = 1
comptime COMMIT_ADAPTIVE = 2

comptime RESIDENCY_SET = 0
comptime RESIDENCY_PER_CB = 1
comptime RESIDENCY_PER_LAUNCH = 2

# MTLResourceStorageModeShared (0) | MTLResourceHazardTrackingModeUntracked
comptime _BUF_OPTIONS = 0 | (1 << 8)
comptime _USAGE_RW = 1 | 2  # MTLResourceUsageRead | MTLResourceUsageWrite


def policy_name(p: Int) -> String:
    if p == COMMIT_EACH:
        return "each"
    if p == COMMIT_EVERY_N:
        return "every-N"
    return "adaptive"


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
    var command_buffer: Int
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
    var use_resources: Int
    var add_allocation: Int
    var rs_commit: Int

    def __init__(out self, o: ObjC):
        self.command_buffer = o.sel("commandBuffer")
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
        self.use_resources = o.sel("useResources:count:usage:")
        self.add_allocation = o.sel("addAllocation:")
        self.rs_commit = o.sel("commit")


struct MetalRT(Movable):
    var o: ObjC
    var s: _Sel
    var device: Int
    var queue: Int
    var event: Int
    var rset: Int
    var policy: Int
    var batch_n: Int
    var residency: Int
    var wait_chain: Bool
    var cb: Int
    var enc: Int
    var pool: Int
    var pending: Int  # launches encoded into the open command buffer
    var committed: Int  # command buffers committed == value the last signals
    var cur_pso: Int
    var rset_dirty: Bool
    var live: List[Int]  # every live MTLBuffer (for the per-cb modes)
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
        wait_chain: Bool = True,
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
        self.policy = policy
        self.batch_n = batch_n
        self.residency = residency
        self.wait_chain = wait_chain
        self.cb = 0
        self.enc = 0
        self.pool = 0
        self.pending = 0
        self.committed = 0
        self.cur_pso = 0
        self.rset_dirty = False
        self.live = List[Int]()
        self._k_keys = List[Int]()
        self._k_pso = List[Int]()
        self._dims = List[Int](length=6, fill=1)
        self.n_launches = 0
        self.n_commits = 0
        self.n_idle_commits = 0
        self.rset = 0
        if residency == RESIDENCY_SET:
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

    def name(self) -> String:
        return self.o.utf8(self.o.send0(self.device, self.o.sel("name")))

    # ---------------------------------------------------------------- buffers

    def alloc[dt: DType](mut self, count: Int) raises -> Buffer:
        var nbytes = max(count * size_of[Scalar[dt]](), 16)
        var obj = self.o.send2(
            self.device,
            self.o.sel("newBufferWithLength:options:"),
            nbytes,
            _BUF_OPTIONS,
        )
        if obj == 0:
            raise Error("newBufferWithLength failed")
        var b = Buffer(
            obj,
            self.o.send0(obj, self.o.sel("contents")),
            self.o.send0(obj, self.o.sel("gpuAddress")),
            nbytes,
        )
        self.live.append(obj)
        if self.rset != 0:
            _ = self.o.send1(self.rset, self.s.add_allocation, obj)
            self.rset_dirty = True
        return b

    def free(mut self, b: Buffer) raises:
        """Release a buffer. Syncs first: the GPU may still be using it."""
        self.sync()
        for i in range(len(self.live)):
            if self.live[i] == b.obj:
                _ = self.live.pop(i)
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
        var pool = self.o.pool_push()
        var data = self.o.system.get_function[Int]("dispatch_data_create")(
            Int(lib_bytes.unsafe_ptr()), lib_bytes.byte_length(), 0, 0
        )
        var err = List[Int](length=1, fill=0)
        var errp = Int(err.unsafe_ptr())
        var mlib = self.o.send2(
            self.device, self.o.sel("newLibraryWithData:error:"), data, errp
        )
        self.o.check_error(err[0], "newLibraryWithData")
        var mfn = self.o.send1(
            mlib, self.o.sel("newFunctionWithName:"), self.o.nsstring(fname)
        )
        if mfn == 0:
            raise Error("function not in metallib: " + fname)
        var pso = self.o.send2(
            self.device,
            self.o.sel("newComputePipelineStateWithFunction:error:"),
            mfn,
            errp,
        )
        self.o.check_error(err[0], "newComputePipelineStateWithFunction")
        self.o.release(mfn)
        self.o.release(mlib)
        self.o.release(data)
        self.o.pool_pop(pool)
        return pso

    # ---------------------------------------------------------------- launch

    def launch[
        func_type: TrivialRegisterPassable, //, func: func_type, *Ts: AnyType
    ](mut self, *args: *Ts, grid: Int, block: Int) raises:
        """Encode one dispatch of `grid` threadgroups of `block` threads.
        Argument i is bound at buffer index i by value (see module doc)."""
        var pso = self._pipeline[func]()
        if self.enc == 0:
            self._begin()
        if pso != self.cur_pso:
            _ = self.o.send1(self.enc, self.s.set_pso, pso)
            self.cur_pso = pso
        comptime for i in range(args.__len__()):
            _ = self.o.send3(
                self.enc,
                self.s.set_bytes,
                Int(Pointer(to=args[i])),
                size_of[Ts[i]](),
                i,
            )
        if self.residency == RESIDENCY_PER_LAUNCH:
            self._use_all()
        self._dims[0] = grid
        self._dims[3] = block
        var d = Int(self._dims.unsafe_ptr())
        _ = self.o.send2(self.enc, self.s.dispatch, d, d + 24)
        self.pending += 1
        self.n_launches += 1
        self._maybe_commit()

    def in_flight(self) -> Int:
        """Command buffers committed but not yet completed."""
        return self.committed - self.o.send0(self.event, self.s.signaled_value)

    def _maybe_commit(mut self):
        if self.policy == COMMIT_EACH:
            self._commit()
        elif self.policy == COMMIT_EVERY_N:
            if self.pending >= self.batch_n:
                self._commit()
        else:
            if self.pending >= self.batch_n:
                self._commit()
            elif self.in_flight() == 0:
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
            _ = self.o.send0(self.rset, self.s.rs_commit)
            self.rset_dirty = False
        # One autorelease pool per command buffer: the command buffer and its
        # encoder are autoreleased; after `commit` the queue owns the buffer.
        self.pool = self.o.pool_push()
        self.cb = self.o.send0(self.queue, self.s.command_buffer_unretained)
        if self.wait_chain and self.committed > 0:
            _ = self.o.send2(self.cb, self.s.wait_event, self.event, self.committed)
        self.enc = self.o.send0(self.cb, self.s.compute_encoder)
        self.cur_pso = 0
        if self.residency == RESIDENCY_PER_CB:
            self._use_all()

    def _commit(mut self):
        if self.cb == 0:
            return
        _ = self.o.send0(self.enc, self.s.end_encoding)
        self.committed += 1
        _ = self.o.send2(self.cb, self.s.signal_event, self.event, self.committed)
        _ = self.o.send0(self.cb, self.s.commit)
        self.o.pool_pop(self.pool)
        self.cb = 0
        self.enc = 0
        self.pool = 0
        self.pending = 0
        self.n_commits += 1

    def flush(mut self):
        """Commit whatever is encoded; do not wait."""
        self._commit()

    def sync(mut self):
        """Commit, then spin until the GPU has finished everything."""
        self._commit()
        while self.o.send0(self.event, self.s.signaled_value) < self.committed:
            pass
