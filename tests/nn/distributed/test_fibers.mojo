"""Gates for `RankFibers` (fibers.mojo).

1. Order: 3 ranks, 4 rendezvous each. The trace must be block-major,
   (0,k) (1,k) (2,k) act(k) ..., with the collective run once per rendezvous,
   by the last rank to arrive.
2. Body state: each rank's body appends to its own `List` and counter on
   both sides of every rendezvous; after the run the host reads every
   entry. One body shared by all ranks lost rank 0's appends: Mojo kept the
   shared `mut self` fields in registers across the rendezvous and the other
   rank wrote its stale copy back (fibers.mojo, the contract).
3. GPU work from the fibers: each rank fills its own device buffer on the
   shared context and the last arriver allreduces them; the host checks the
   sums after the run.
4. A rank that raises ends the run with its error (no hang).
5. A rank with one rendezvous fewer than the others ends the run with a
   mismatch error (no hang).

Run:  pixi run -e apple mojo build -I . tests/nn/distributed/test_fibers.mojo -o $B/tf && $B/tf
"""

from std.sys import has_accelerator
from std.testing import assert_true, assert_equal
from std.memory import Pointer
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.fill import fill_dev
from noeira.core.concurrent.block import ControlBlock, ControlBlockView
from noeira.nn.distributed.fibers import (
    RankFibers,
    RankStep,
    fiber_arrive,
    fiber_depart,
    fiber_current,
)
from noeira.nn.distributed.process_group import ProcessGroup


comptime NR = 3
comptime K = 4
comptime ACT = 100


struct _Trace(RankStep):
    """Logs (rank, point) as 10*k + rank into a shared calloc'ed log, and
    ACT + k for a collective, with the acting rank in a second log."""

    var log: Int
    var who: Int
    var bad_rank: Int
    var short_rank: Int

    def __init__(out self, log: Int, who: Int, bad_rank: Int, short_rank: Int):
        self.log = log
        self.who = who
        self.bad_rank = bad_rank
        self.short_rank = short_rank

    def _push(self, x: Int):
        var v = ControlBlockView(self.log)
        var i = Int(v.fetch_add(0, 1))
        v.release_store(1 + i, Int64(x))

    def run_rank(mut self, r: Int) raises:
        assert_equal(fiber_current(), r)
        var kk = K - 1 if r == self.short_rank else K
        for k in range(kk):
            self._push(10 * k + r)
            if r == self.bad_rank and k == 2:
                raise Error("boom")
            if fiber_arrive(r):
                self._push(ACT + k)
                ControlBlockView(self.who).release_store(k, Int64(r))
                fiber_depart(r)


def _traces(log: ControlBlock, who: ControlBlock, bad: Int, short: Int) -> List[_Trace]:
    var out = List[_Trace]()
    for _ in range(NR):
        out.append(_Trace(log.addr(), who.addr(), bad, short))
    return out^


def _test_order() raises:
    var log = ControlBlock(256)
    var who = ControlBlock(64)
    var bodies = _traces(log, who, -1, -1)
    RankFibers.run(bodies)
    var v = log.view()
    var got = List[Int]()
    for i in range(Int(v.acquire_load(0))):
        got.append(Int(v.acquire_load(1 + i)))
    var want = List[Int]()
    for k in range(K):
        for r in range(NR):
            want.append(10 * k + r)
        want.append(ACT + k)
    if len(got) != len(want):
        var t = String("  trace:")
        for i in range(len(got)):
            t += " " + String(got[i])
        print(t)
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_equal(got[i], want[i])
    for k in range(K):
        assert_equal(Int(who.view().acquire_load(k)), NR - 1)
    # The bodies hold only addresses: the owners must outlive every read
    # (a `view()` is otherwise the owner's last use, and frees it).
    _ = log^
    _ = who^
    print("  order: block-major, one collective per rendezvous by the last rank")


struct _State(RankStep):
    """Per-rank Mojo state written on both sides of every rendezvous."""

    var seen: List[Int]
    var count: Int

    def __init__(out self):
        self.seen = List[Int]()
        self.count = 0

    def run_rank(mut self, r: Int) raises:
        for k in range(K):
            self.seen.append(10 * k + r)
            self.count += 1
            if fiber_arrive(r):
                fiber_depart(r)
            self.count += 100


def _test_state() raises:
    var bodies = List[_State]()
    for _ in range(NR):
        bodies.append(_State())
    for _ in range(2):
        RankFibers.run(bodies)
    for r in range(NR):
        assert_equal(len(bodies[r].seen), 2 * K)
        assert_equal(bodies[r].count, 2 * K * 101)
        for i in range(2 * K):
            assert_equal(bodies[r].seen[i], 10 * (i % K) + r)
    print("  body state: every rank's appends and counts survive the run")


struct _GpuShared(Movable):
    var pg: ProcessGroup[NR]
    var buf: List[Tensor]
    var acc: List[Tensor]

    def __init__(out self, ctx: DeviceContext) raises:
        self.pg = ProcessGroup[NR].shared(ctx)
        self.buf = List[Tensor]()
        self.acc = List[Tensor]()
        for _ in range(NR):
            self.buf.append(Tensor.alloc_gpu(ctx, 64))
            self.acc.append(Tensor.alloc_gpu(ctx, 64))


struct _GpuStep(RankStep):
    """Rank r fills buf[r] with r + 1 + 10k; the last arriver allreduces
    every rank's buffer into acc."""

    var shared: Int

    def __init__(out self, shared: Int):
        self.shared = shared

    def run_rank(mut self, r: Int) raises:
        var sp = Pointer[_GpuShared, MutUntrackedOrigin](
            unsafe_from_address=self.shared
        )
        for k in range(K):
            fill_dev(
                sp[].buf[r].dev.value(), 64, Scalar[DT](r + 1 + 10 * k),
                sp[].pg.ctx(r),
            )
            if fiber_arrive(r):
                var ins = List[DeviceBuffer[DT]]()
                var outs = List[DeviceBuffer[DT]]()
                for q in range(NR):
                    ins.append(sp[].buf[q].dev.value())
                    outs.append(sp[].acc[q].dev.value())
                sp[].pg.allreduce_sum(ins, outs, 64)
                fiber_depart(r)


def _test_gpu(ctx: DeviceContext) raises:
    var s = _GpuShared(ctx)
    var bodies = List[_GpuStep]()
    for _ in range(NR):
        bodies.append(_GpuStep(Int(Pointer(to=s))))
    RankFibers.run(bodies)
    s.pg.synchronize()
    # last rendezvous: buf[q] = q + 1 + 30 -> sum = 1 + 2 + 3 + 90 = 96
    for q in range(NR):
        s.acc[q].download(ctx)
        for i in range(64):
            assert_equal(Float64(s.acc[q].data[i]), 96.0)
    print("  gpu: fills and an allreduce enqueued from fibers, sums correct")


def _test_error() raises:
    var log = ControlBlock(256)
    var who = ControlBlock(64)
    var bodies = _traces(log, who, 1, -1)
    var raised = False
    try:
        RankFibers.run(bodies)
    except e:
        raised = True
        print("  error: raised as expected:", e)
        assert_true("boom" in String(e), "the rank's own error must surface")
    assert_true(raised, "a failing rank must fail the run")
    _ = log^
    _ = who^


def _test_mismatch() raises:
    var log = ControlBlock(256)
    var who = ControlBlock(64)
    var bodies = _traces(log, who, -1, 0)
    var raised = False
    try:
        RankFibers.run(bodies)
    except e:
        raised = True
        print("  mismatch: raised as expected:", e)
    assert_true(raised, "a rank with fewer rendezvous must fail the run")
    _ = log^
    _ = who^


def main() raises:
    print("RankFibers gates:", NR, "ranks,", K, "rendezvous")
    comptime if not has_accelerator():
        print("No accelerator — skipping (the distributed gates need a GPU, or Metal for the simulator)")
        return
    _test_order()
    _test_state()
    _test_gpu(DeviceContext())
    _test_error()
    _test_mismatch()
    # the scheduler is reusable after failures
    _test_order()
    print("FIBERS GATES OK")
