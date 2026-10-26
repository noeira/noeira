"""The same launcher on the Metal 4 API (macOS 26+, MTLGPUFamilyMetal4; the
M1 Pro reports it). Same `launch[kernel](args..., grid, block)` / `sync()` /
`flush()` / commit policies as `MetalRT` (`Metal4RT.alloc` returns the same
`Buffer`).

Written from MTL4CommandQueue.h, MTL4CommandBuffer.h, MTL4CommandAllocator.h,
MTL4ArgumentTable.h, MTL4ComputeCommandEncoder.h and MTL4CommandEncoder.h.
What differs from the classic path:

- No `setBytes`. An `MTL4ArgumentTable` holds GPU ADDRESSES only, so each
  argument's bytes are written into a shared "argument ring" buffer and the
  table slot i gets the address of argument i's bytes. "Metal takes a
  snapshot of the resources in the argument table when you make dispatch
  calls", so one table is reused for every dispatch.
- Compute encoders are concurrent: dependent dispatches need an intra-pass
  barrier (`barrierAfterEncoderStages:Dispatch beforeEncoderStages:Dispatch`)
  between them, and each encoder starts with a queue barrier
  (`barrierAfterQueueStages:Dispatch beforeStages:Dispatch`) to order it
  after earlier command buffers.
- Command buffers and allocators are long-lived, reusable objects. An
  allocator may only be reset once every command buffer that used it is
  complete; the queue signals a shared event after each commit
  (`signalEvent:value:`) and a ring of allocators is recycled by that value.
- Residency is a residency set attached to the queue (required: there is no
  `useResource` in Metal 4).
"""

from std.memory import Pointer
from std.sys import size_of
from std.time import perf_counter_ns
from max.gpu.host.compile import _compile_code

from noeira.experimental.metal_rt.objc import ObjC
from noeira.experimental.metal_rt.runtime import (
    Buffer,
    Launcher,
    load_pipeline,
    COMMIT_EACH,
    COMMIT_EVERY_N,
    COMMIT_ADAPTIVE,
)

comptime _STAGE_DISPATCH = 1 << 27  # MTLStageDispatch
comptime _VIS_DEVICE = 1 << 0  # MTL4VisibilityOptionDevice
comptime _NRING = 16  # command buffers / allocators in flight at most
comptime _ARG_RING_BYTES = 4 << 20
comptime _MAX_ARGS = 31


struct Metal4RT(Launcher):
    var o: ObjC
    var device: Int
    var queue: Int
    var event: Int
    var rset: Int
    var argtab: Int
    var ring: Buffer  # argument bytes
    var ring_off: Int
    var cbs: List[Int]  # MTL4CommandBuffer objects, one per ring slot
    var allocs: List[Int]  # MTL4CommandAllocator objects
    var policy: Int
    var batch_n: Int
    var idle_age_ns: Int
    var first_pending_ns: Int
    var barriers: Bool  # False = no intra-pass / queue barriers (UNSAFE; measurement)
    var cb: Int
    var enc: Int
    var pool: Int
    var pending: Int
    var committed: Int
    var cur_pso: Int
    var rset_dirty: Bool
    var _k_keys: List[Int]
    var _k_pso: List[Int]
    var _dims: List[Int]
    var _cbp: List[Int]  # one-element array for commit:count:
    # selectors
    var s_begin: Int
    var s_end_cb: Int
    var s_enc: Int
    var s_set_pso: Int
    var s_set_table: Int
    var s_set_addr: Int
    var s_dispatch: Int
    var s_barrier_enc: Int
    var s_barrier_queue: Int
    var s_end_enc: Int
    var s_commit: Int
    var s_signal: Int
    var s_signaled: Int
    var s_reset: Int
    var s_rs_commit: Int
    var s_add_alloc: Int
    var n_launches: Int
    var n_commits: Int
    var n_idle_commits: Int

    def __init__(
        out self,
        policy: Int = COMMIT_ADAPTIVE,
        batch_n: Int = 16,
        idle_age_us: Int = 0,
        barriers: Bool = True,
    ) raises:
        self.o = ObjC()
        self.device = self.o.metal.get_function[Int](
            "MTLCreateSystemDefaultDevice"
        )()
        if not self.o.send_bool(self.device, self.o.sel("supportsFamily:"), 5002):
            raise Error("device does not support MTLGPUFamilyMetal4")
        self.queue = self.o.send0(self.device, self.o.sel("newMTL4CommandQueue"))
        if self.queue == 0:
            raise Error("newMTL4CommandQueue failed")
        self.event = self.o.send0(self.device, self.o.sel("newSharedEvent"))

        # residency set on the queue
        var desc = self.o.send0(
            self.o.send0(self.o.cls("MTLResidencySetDescriptor"), self.o.sel("alloc")),
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

        # argument table
        var tdesc = self.o.send0(
            self.o.send0(self.o.cls("MTL4ArgumentTableDescriptor"), self.o.sel("alloc")),
            self.o.sel("init"),
        )
        _ = self.o.send1(tdesc, self.o.sel("setMaxBufferBindCount:"), _MAX_ARGS)
        self.argtab = self.o.send2(
            self.device,
            self.o.sel("newArgumentTableWithDescriptor:error:"),
            tdesc,
            Int(err.unsafe_ptr()),
        )
        self.o.check_error(err[0], "newArgumentTableWithDescriptor")
        self.o.release(tdesc)

        self.cbs = List[Int]()
        self.allocs = List[Int]()
        for _ in range(_NRING):
            self.cbs.append(self.o.send0(self.device, self.o.sel("newCommandBuffer")))
            self.allocs.append(
                self.o.send0(self.device, self.o.sel("newCommandAllocator"))
            )
        self.policy = policy
        self.batch_n = batch_n
        self.idle_age_ns = idle_age_us * 1000
        self.first_pending_ns = 0
        self.barriers = barriers
        self.cb = 0
        self.enc = 0
        self.pool = 0
        self.pending = 0
        self.committed = 0
        self.cur_pso = 0
        self.rset_dirty = False
        self._k_keys = List[Int]()
        self._k_pso = List[Int]()
        self._dims = List[Int](length=6, fill=1)
        self._cbp = List[Int](length=1, fill=0)
        self.s_begin = self.o.sel("beginCommandBufferWithAllocator:")
        self.s_end_cb = self.o.sel("endCommandBuffer")
        self.s_enc = self.o.sel("computeCommandEncoder")
        self.s_set_pso = self.o.sel("setComputePipelineState:")
        self.s_set_table = self.o.sel("setArgumentTable:")
        self.s_set_addr = self.o.sel("setAddress:atIndex:")
        self.s_dispatch = self.o.sel("dispatchThreadgroups:threadsPerThreadgroup:")
        self.s_barrier_enc = self.o.sel(
            "barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:"
        )
        self.s_barrier_queue = self.o.sel(
            "barrierAfterQueueStages:beforeStages:visibilityOptions:"
        )
        self.s_end_enc = self.o.sel("endEncoding")
        self.s_commit = self.o.sel("commit:count:")
        self.s_signal = self.o.sel("signalEvent:value:")
        self.s_signaled = self.o.sel("signaledValue")
        self.s_reset = self.o.sel("reset")
        self.s_rs_commit = self.o.sel("commit")
        self.s_add_alloc = self.o.sel("addAllocation:")
        self.n_launches = 0
        self.n_commits = 0
        self.n_idle_commits = 0
        self.ring = Buffer(0, 0, 0, 0)
        self.ring_off = 0
        self.ring = self.alloc[DType.uint8](_ARG_RING_BYTES)

    def commits(self) -> Int:
        return self.n_commits

    def idle_commits(self) -> Int:
        return self.n_idle_commits

    def alloc[dt: DType](mut self, count: Int) raises -> Buffer:
        var nbytes = max(count * size_of[Scalar[dt]](), 16)
        var obj = self.o.send2(
            self.device,
            self.o.sel("newBufferWithLength:options:"),
            nbytes,
            0 | (1 << 8),  # shared, untracked
        )
        if obj == 0:
            raise Error("newBufferWithLength failed")
        var b = Buffer(
            obj,
            self.o.send0(obj, self.o.sel("contents")),
            self.o.send0(obj, self.o.sel("gpuAddress")),
            nbytes,
        )
        _ = self.o.send1(self.rset, self.s_add_alloc, obj)
        self.rset_dirty = True
        return b

    def _pipeline[func_type: TrivialRegisterPassable, //, func: func_type](mut self) raises -> Int:
        var info = _compile_code[func, emission_kind="object"]()
        var fname = info.function_name
        var key = Int(fname.unsafe_ptr())
        for i in range(len(self._k_keys)):
            if self._k_keys[i] == key:
                return self._k_pso[i]
        var pso = load_pipeline(self.o, self.device, String(info.asm), String(fname))
        self._k_keys.append(key)
        self._k_pso.append(pso)
        return pso

    def signaled(self) -> Int:
        return self.o.send0(self.event, self.s_signaled)

    def gpu_idle(self) -> Bool:
        return self.signaled() >= self.committed

    def launch[
        func_type: TrivialRegisterPassable, //, func: func_type, *Ts: AnyType
    ](mut self, *args: *Ts, grid: Int, block: Int) raises:
        var pso = self._pipeline[func]()
        # argument bytes, 16-byte aligned per argument
        var need = 0
        comptime for i in range(args.__len__()):
            need += (size_of[Ts[i]]() + 15) & ~15
        if self.ring_off + need > _ARG_RING_BYTES:
            # wrap: everything that read the ring must be done first
            self.sync()
            self.ring_off = 0
        if self.enc == 0:
            self._begin()
            if self.idle_age_ns > 0:
                self.first_pending_ns = Int(perf_counter_ns())
        elif self.barriers:
            # concurrent encoder: order this dispatch after the previous one
            _ = self.o.send3(
                self.enc, self.s_barrier_enc, _STAGE_DISPATCH, _STAGE_DISPATCH, _VIS_DEVICE
            )
        if pso != self.cur_pso:
            _ = self.o.send1(self.enc, self.s_set_pso, pso)
            self.cur_pso = pso
        comptime for i in range(args.__len__()):
            comptime sz = size_of[Ts[i]]()
            var dst = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=self.ring.host + self.ring_off)
            var src = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(Pointer(to=args[i])))
            comptime for k in range(sz):
                dst[unsafe_offset=k] = src[unsafe_offset=k]
            _ = self.o.send2(self.argtab, self.s_set_addr, self.ring.gpu + self.ring_off, i)
            self.ring_off += (sz + 15) & ~15
        self._dims[0] = grid
        self._dims[3] = block
        var d = Int(self._dims.unsafe_ptr())
        _ = self.o.send2(self.enc, self.s_dispatch, d, d + 24)
        self.pending += 1
        self.n_launches += 1
        if self.policy == COMMIT_EACH:
            self._commit()
        elif self.pending >= self.batch_n:
            self._commit()
        elif self.policy == COMMIT_ADAPTIVE and (
            self.idle_age_ns == 0
            or Int(perf_counter_ns()) - self.first_pending_ns >= self.idle_age_ns
        ) and self.gpu_idle():
            self.n_idle_commits += 1
            self._commit()

    def _begin(mut self):
        if self.rset_dirty:
            _ = self.o.send0(self.rset, self.s_rs_commit)
            self.rset_dirty = False
        var slot = self.committed % _NRING
        # The allocator of this slot was last used by command buffer
        # (committed + 1 - _NRING); it must be complete before reset.
        var need = self.committed + 1 - _NRING
        if need > 0:
            while self.signaled() < need:
                pass
        _ = self.o.send0(self.allocs[slot], self.s_reset)
        # Retain the (autoreleased) encoder; no pool may span an open
        # command buffer (see runtime.mojo `_begin`).
        var pool = self.o.pool_push()
        self.cb = self.cbs[slot]
        _ = self.o.send1(self.cb, self.s_begin, self.allocs[slot])
        self.enc = self.o.send0(self.o.send0(self.cb, self.s_enc), self.o.sel("retain"))
        self.o.pool_pop(pool)
        if self.committed > 0 and self.barriers:
            _ = self.o.send3(
                self.enc, self.s_barrier_queue, _STAGE_DISPATCH, _STAGE_DISPATCH, _VIS_DEVICE
            )
        # Set once per encoder: the encoder snapshots the table's contents at
        # each dispatch, so later setAddress calls still reach later dispatches.
        _ = self.o.send1(self.enc, self.s_set_table, self.argtab)
        self.cur_pso = 0

    def _commit(mut self):
        if self.cb == 0:
            return
        var pool = self.o.pool_push()
        _ = self.o.send0(self.enc, self.s_end_enc)
        self.o.release(self.enc)
        _ = self.o.send0(self.cb, self.s_end_cb)
        self._cbp[0] = self.cb
        _ = self.o.send2(self.queue, self.s_commit, Int(self._cbp.unsafe_ptr()), 1)
        self.committed += 1
        _ = self.o.send2(self.queue, self.s_signal, self.event, self.committed)
        self.o.pool_pop(pool)
        self.cb = 0
        self.enc = 0
        self.pool = 0
        self.pending = 0
        self.n_commits += 1

    def flush(mut self):
        self._commit()

    def sync(mut self):
        self._commit()
        while self.signaled() < self.committed:
            pass
