"""RankFibers — every rank's step on its own OS thread, ONE thread running at a time.

Why it exists. MAX's collectives are driven from one host thread, which
enqueues every rank's half of a collective itself; noeira's backward is one
composed `vjp` call per replica. So one thread enqueues rank 0's WHOLE
backward before rank 1's starts. A collective over one block (ZeRO-2's
reduce-scatter of its gradients, ZeRO-3's all-gather of its weights) needs
every rank's input for that block enqueued before it, and on the naive and
simulator backends it must be enqueued AFTER those inputs: they order ranks
with events, and an event wait cannot target work the host has not enqueued
yet. Freeing a block's gradient buffer before the next block's vjp needs the
same thing. In rank-major order neither is possible.

So each rank's step runs on its own thread, as one process per rank would
(torchrun), but the threads pass a baton: exactly one enqueues at any time.
Nothing in MAX or noeira has to be thread-safe, and the enqueue order is
deterministic. A rank yields at a rendezvous; the LAST rank to arrive runs the
collective for every rank from its own thread, through the same single-thread
`ProcessGroup` calls as everywhere else, then passes the baton on:

    if fiber_arrive(r):          # True on the last rank to arrive
        <collective for all ranks>
        fiber_depart(r)

The enqueue order becomes block-major:

    R0 vjp(blk 5) | R1 vjp(blk 5) | RS(blk 5) | R0 vjp(blk 4) | R1 vjp(blk 4) | ...

These are coroutines built from threads: Mojo has no coroutine that can
suspend in the middle of a composed `vjp`.

Contract
- ONE BODY PER RANK (`run(bodies)`), as one process per rank would have.
  Never let two fibers hold a `mut` reference to the same object across a
  rendezvous. Mojo treats `mut` as exclusive, so a fiber suspended inside
  `run_rank(mut self)` may keep `self`'s fields in registers and store them
  back when it returns, over whatever another fiber wrote meanwhile. That
  happened with one shared body: rank 1 resumed after the last rendezvous
  and wrote back its stale empty `losses`, erasing rank 0's appends (Mac,
  2026-10-09). State shared between ranks (the driver, the replicas) is
  reached through an address (`Pointer(unsafe_from_address=...)`), with each
  borrow taken and dropped between two rendezvous; a callee that reaches a
  rendezvous must not be holding a `mut` borrow of shared state.
- Every rank must reach the same rendezvous, in the same order. A rank that
  finishes its step while others wait, or one that arrives after another
  finished, fails the run with an error instead of hanging.
- A rank's error ends the run: the other ranks raise from their next (or
  current) rendezvous, every thread is joined, and `run` raises the first
  error.
- Cross-thread scalars live in a `calloc`ed `ControlBlock`
  (`noeira/core/concurrent/block.mojo` explains why: a Mojo-owned slab read
  after a join was folded to its initializer). The baton is a release store
  / acquire load, so everything one fiber wrote is visible to the next.
- Threads get a 64 MB stack: macOS gives a secondary thread 512 KB, and a
  composed model's vjp is a deep chain of large frames.
"""

from std.ffi import _get_global_or_null, external_call, c_int
from std.memory import Pointer
from std.benchmark import keep
from std.memory.alloc import Layout as AllocLayout

from noeira.core.concurrent.block import ControlBlock, ControlBlockView
from noeira.core.concurrent.thread import (
    OpaquePtr,
    ThreadHandle,
    null_opaque,
    opaque_from_address,
)


comptime MAX_FIBERS = 64
comptime FIBER_STACK_BYTES = 64 << 20

# ── control cells ────────────────────────────────────────────────────────────
comptime _C_TURN = 0
"""Rank allowed to run; -1 when every rank has finished."""
comptime _C_ARRIVED = 1
"""Ranks waiting at the current rendezvous (only the running fiber writes)."""
comptime _C_DONE = 2
"""Ranks whose step has returned (or failed)."""
comptime _C_FAILED = 3
comptime _C_N = 4
comptime _C_SEQ = 5
"""Completed rendezvous. A waiting rank checks it moved before trusting that
its rendezvous ran (it did not if another rank finished instead)."""
comptime _C_ERR_LEN = 6
comptime _C_RANK_DONE = 8
comptime _C_ERR = _C_RANK_DONE + MAX_FIBERS
comptime _ERR_BYTES = 1024
comptime _N_CELLS = _C_ERR + _ERR_BYTES // 8

comptime _SPAWN_TID = 24
"""Cell of `_spawn`'s block that receives the `pthread_t`; cells below it hold
the `pthread_attr_t` (56 B on Linux, 64 B on macOS)."""
comptime _SPAWN_CELLS = 32


trait RankStep:
    """One rank's part of a step: enqueue its forward, loss and backward.
    One instance per rank (see the contract above)."""

    def run_rank(mut self, r: Int) raises:
        ...


# ── baton ────────────────────────────────────────────────────────────────────


@always_inline
def _yield_cpu():
    _ = external_call["sched_yield", Int32]()


def _wait_turn(v: ControlBlockView, r: Int):
    while Int(v.acquire_load(_C_TURN)) != r:
        _yield_cpu()


def _pass(v: ControlBlockView, from_r: Int):
    """Hand the baton to the next rank (round robin) still running."""
    var n = Int(v.acquire_load(_C_N))
    for i in range(1, n + 1):
        var k = (from_r + i) % n
        if v.acquire_load(_C_RANK_DONE + k) == 0:
            v.release_store(_C_TURN, Int64(k))
            return
    v.release_store(_C_TURN, -1)


def _fail(v: ControlBlockView, msg: String):
    """Record the run's first error; later ones are dropped."""
    if v.acquire_load(_C_FAILED) != 0:
        return
    var dst = Pointer[UInt8, MutUntrackedOrigin](
        unsafe_from_address=v.addr + _C_ERR * 8
    )
    var src = msg.as_bytes()
    var n = min(len(src), _ERR_BYTES)
    for i in range(n):
        dst.unsafe_offset(i)[] = src[i]
    v.release_store(_C_ERR_LEN, Int64(n))
    v.release_store(_C_FAILED, 1)


def _error_text(v: ControlBlockView) -> String:
    var n = Int(v.acquire_load(_C_ERR_LEN))
    var p = Pointer[UInt8, MutUntrackedOrigin](
        unsafe_from_address=v.addr + _C_ERR * 8
    )
    var out = String()
    for i in range(n):
        out += chr(Int(p.unsafe_offset(i)[]))
    return out^


# ── the active run, process-wide ─────────────────────────────────────────────

comptime _ACTIVE = "NOEIRA_RANK_FIBERS"


struct _ActiveRun(Movable):
    var addr: Int
    """The running `RankFibers`' control block; 0 when none runs."""

    def __init__(out self):
        self.addr = 0


def _active() raises -> Pointer[_ActiveRun, UntrackedOrigin[mut=True]]:
    var g = _get_global_or_null(_ACTIVE)
    if not g:
        var p = alloc(AllocLayout[_ActiveRun].single()).unsafe_leak()
        p.unsafe_write(_ActiveRun())
        external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
            StringSlice(_ACTIVE), p.unsafe_bitcast[NoneType]()
        )
        g = _get_global_or_null(_ACTIVE)
    return g.value().unsafe_bitcast[_ActiveRun]()


def fibers_active() raises -> Bool:
    """Whether a `RankFibers.run` is in progress (called from its fibers)."""
    return _active()[].addr != 0


def fiber_current() raises -> Int:
    """The rank holding the baton, or -1 outside a run."""
    var a = _active()[].addr
    if a == 0:
        return -1
    return Int(ControlBlockView(a).acquire_load(_C_TURN))


def fiber_arrive(r: Int) raises -> Bool:
    """Rank r reaches a rendezvous. Returns True on the last rank to arrive,
    which must then run the collective for every rank and call
    `fiber_depart(r)`. Every other rank yields here and returns False once
    the collective has been enqueued. Outside a run: True (one rank)."""
    var a = _active()[].addr
    if a == 0:
        return True
    var v = ControlBlockView(a)
    if v.acquire_load(_C_FAILED) != 0:
        raise Error("another rank failed")
    if Int(v.acquire_load(_C_TURN)) != r:
        raise Error(
            "fiber_arrive: rank " + String(r) + " does not hold the baton"
        )
    if v.acquire_load(_C_DONE) != 0:
        _fail(
            v,
            "rank " + String(r)
            + " reached a rendezvous after another rank finished its step"
            " (ranks ran different collective sequences)",
        )
        raise Error("rendezvous mismatch")
    var n = Int(v.acquire_load(_C_N))
    var arrived = Int(v.acquire_load(_C_ARRIVED)) + 1
    if arrived == n:
        v.release_store(_C_ARRIVED, 0)
        return True
    v.release_store(_C_ARRIVED, Int64(arrived))
    var seq = v.acquire_load(_C_SEQ)
    _pass(v, r)
    _wait_turn(v, r)
    if v.acquire_load(_C_FAILED) != 0:
        raise Error("another rank failed")
    if v.acquire_load(_C_SEQ) == seq:
        _fail(v, "rank " + String(r) + "'s rendezvous never completed")
        raise Error("rendezvous mismatch")
    return False


def fiber_depart(r: Int) raises:
    """The last arriver, after enqueueing the collective: let the other ranks
    resume (in rank order after r), and wait for the baton again."""
    var a = _active()[].addr
    if a == 0:
        return
    var v = ControlBlockView(a)
    _ = v.fetch_add(_C_SEQ, 1)
    _pass(v, r)
    _wait_turn(v, r)
    if v.acquire_load(_C_FAILED) != 0:
        raise Error("another rank failed")


# ── threads ──────────────────────────────────────────────────────────────────


struct _FiberArg(Copyable, Movable):
    var body: Int
    var blk: Int
    var rank: Int

    def __init__(out self, body: Int, blk: Int, rank: Int):
        self.body = body
        self.blk = blk
        self.rank = rank


def _fiber_entry[B: RankStep](arg: OpaquePtr) -> OpaquePtr:
    """Thread entry, one per body type. Must not raise."""
    var a = Pointer[_FiberArg, MutUntrackedOrigin](unsafe_from_address=Int(arg))
    var v = ControlBlockView(a[].blk)
    var r = a[].rank
    var body = Pointer[B, MutUntrackedOrigin](unsafe_from_address=a[].body)
    _wait_turn(v, r)
    if v.acquire_load(_C_FAILED) == 0:
        try:
            body[].run_rank(r)
            if v.acquire_load(_C_ARRIVED) != 0:
                _fail(
                    v,
                    "rank " + String(r)
                    + " finished its step while other ranks wait at a"
                    " rendezvous (ranks ran different collective sequences)",
                )
        except e:
            _fail(v, "rank " + String(r) + ": " + String(e))
    v.release_store(_C_RANK_DONE + r, 1)
    _ = v.fetch_add(_C_DONE, 1)
    _pass(v, r)
    return null_opaque()


def _spawn[
    start: def(OpaquePtr) thin -> OpaquePtr
](arg: OpaquePtr, stack_bytes: Int) raises -> ThreadHandle:
    """`pthread_create` with an explicit stack size.

    The attribute block and the `pthread_t` it writes back live in `calloc`ed
    cells: a `pthread_t` in a Mojo local, written through an address handed
    out as an integer, can be read back as its initializer (block.mojo), and
    a handle of 0 makes `join` return without waiting."""
    var blk = ControlBlock(_SPAWN_CELLS)
    var attr = opaque_from_address(blk.addr())
    var rc = external_call["pthread_attr_init", c_int, OpaquePtr](attr)
    if rc != c_int(0):
        raise Error("pthread_attr_init failed, rc=" + String(Int(rc)))
    rc = external_call["pthread_attr_setstacksize", c_int, OpaquePtr, Int](
        attr, stack_bytes
    )
    if rc != c_int(0):
        raise Error("pthread_attr_setstacksize failed, rc=" + String(Int(rc)))
    var tid_ptr = Pointer[UInt64, MutUntrackedOrigin](
        unsafe_from_address=blk.addr() + _SPAWN_TID * 8
    )
    rc = external_call[
        "pthread_create",
        c_int,
        Pointer[UInt64, MutUntrackedOrigin],
        OpaquePtr,
        def(OpaquePtr) thin -> OpaquePtr,
        OpaquePtr,
    ](tid_ptr, attr, start, arg)
    _ = external_call["pthread_attr_destroy", c_int, OpaquePtr](attr)
    if rc != c_int(0):
        raise Error("pthread_create failed, rc=" + String(Int(rc)))
    var tid = UInt64(blk.view().acquire_load(_SPAWN_TID))
    _ = blk^
    if tid == 0:
        raise Error("pthread_create returned a null thread handle")
    return ThreadHandle(tid)


struct RankFibers:
    """`run(bodies)`: `bodies[r].run_rank(r)` for every rank r, each on its
    own thread, rank 0 first, switching only at rendezvous."""

    @staticmethod
    def run[B: RankStep](mut bodies: List[B]) raises:
        var n = len(bodies)
        if n < 1 or n > MAX_FIBERS:
            raise Error("RankFibers.run: " + String(n) + " ranks")
        var act = _active()
        if act[].addr != 0:
            raise Error("RankFibers.run: a run is already in progress")
        var blk = ControlBlock(_N_CELLS)
        var v = blk.view()
        v.release_store(_C_N, Int64(n))
        v.release_store(_C_TURN, 0)
        act[].addr = blk.addr()
        var args = List[_FiberArg](capacity=n)
        for r in range(n):
            args.append(_FiberArg(Int(Pointer(to=bodies[r])), blk.addr(), r))
        # The fibers reach the bodies through addresses handed to pthread as
        # integers. `keep` passes the list to an opaque asm that may read and
        # write memory, before the spawn and after the joins, so the caller
        # neither keeps a body's fields cached across `run` nor drops a store
        # it made before.
        keep(bodies)
        var threads = List[ThreadHandle](capacity=n)
        for r in range(n):
            try:
                threads.append(
                    _spawn[_fiber_entry[B]](
                        opaque_from_address(Int(Pointer(to=args[r]))),
                        FIBER_STACK_BYTES,
                    )
                )
            except e:
                # The missing ranks count as done, so the spawned ones are
                # not left waiting for them.
                _fail(v, "spawning rank " + String(r) + ": " + String(e))
                for k in range(r, n):
                    v.release_store(_C_RANK_DONE + k, 1)
                    _ = v.fetch_add(_C_DONE, 1)
                if r == 0:
                    v.release_store(_C_TURN, -1)
                break
        for i in range(len(threads)):
            threads[i].join()
        keep(bodies)
        act[].addr = 0
        if v.acquire_load(_C_FAILED) != 0:
            var msg = _error_text(v)
            _ = blk^
            raise Error("RankFibers: " + msg)
        _ = args^
        _ = blk^
