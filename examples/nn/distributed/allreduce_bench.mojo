"""M0 spike: MAX `comm` allreduce through `ProcessGroup`, from one host thread.

For each size: correctness (rank r contributes r + 1, every output must be
N(N+1)/2), then the time per allreduce and the bus bandwidth
`2 (N-1)/N * bytes / t` (the nccl-tests convention, comparable with
`all_reduce_perf`). Then the in-place question: the same allreduce with the
output aliasing the input, checked for correctness at every size (the 1-stage
kernel is expected to race, the 2-stage one not).

Every allreduce is enqueued rank by rank from this thread, so a run that
finishes at all answers "does serial single-thread enqueue deadlock?".

Run on an N-GPU box (N in 2, 4, 8; default 2):
    pixi run -e nvidia mojo build -I . -D NGPUS=2 \
        examples/nn/distributed/allreduce_bench.mojo -o /tmp/arb && /tmp/arb
`-D ARENA=<elems>` adds the model arena size to the sweep (the GPT 6x384
arena prints as `arena =` from the DDP GPT example).
"""

from std.sys import get_defined_int
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer

from noeira.nn.constants import DT
from noeira.nn.distributed.process_group import ProcessGroup, backend_name


comptime N = get_defined_int["NGPUS", 2]()
comptime ARENA = get_defined_int["ARENA", 0]()
comptime ITERS = 20
comptime WARMUP = 3


def _fill(
    mut pg: ProcessGroup[N], bufs: List[DeviceBuffer[DT]], n: Int
) raises:
    for r in range(N):
        var h = List[Scalar[DT]](length=n, fill=Scalar[DT](r + 1))
        pg.ctx(r).enqueue_copy(bufs[r], h.unsafe_ptr())
        pg.ctx(r).synchronize()


def _check(
    mut pg: ProcessGroup[N], outs: List[DeviceBuffer[DT]], n: Int
) raises -> Bool:
    var want = Scalar[DT](N * (N + 1) // 2)
    for r in range(N):
        var h = List[Scalar[DT]](length=n, fill=0)
        pg.ctx(r).enqueue_copy(h.unsafe_ptr(), outs[r])
        pg.ctx(r).synchronize()
        for i in range(n):
            if h[i] != want:
                print(
                    "      rank", r, "elem", i, "=", h[i], "want", want
                )
                return False
    return True


def main() raises:
    var sizes: List[Int] = [1 << 10, 1 << 16, 1 << 20, 1 << 24, 100 << 20]
    if ARENA > 0:
        sizes.append(ARENA)
    var maxn = 0
    for s in sizes:
        maxn = max(maxn, s)
    var pg = ProcessGroup[N].devices(maxn)
    print("allreduce bench: N =", N, " backend =", backend_name(pg.backend))

    var ins = List[DeviceBuffer[DT]]()
    var outs = List[DeviceBuffer[DT]]()
    for r in range(N):
        ins.append(pg.ctx(r).enqueue_create_buffer[DT](maxn))
        outs.append(pg.ctx(r).enqueue_create_buffer[DT](maxn))
    pg.synchronize()

    print("  elems\tbytes\tok\tus/op\tbusbw GB/s\tin-place ok")
    for s in sizes:
        var n = s
        var sub_in = List[DeviceBuffer[DT]]()
        var sub_out = List[DeviceBuffer[DT]]()
        for r in range(N):
            sub_in.append(ins[r].create_sub_buffer[DT](0, n))
            sub_out.append(outs[r].create_sub_buffer[DT](0, n))

        _fill(pg, sub_in, n)
        pg.allreduce_sum(sub_in, sub_out, n)
        pg.synchronize()
        var ok = _check(pg, sub_out, n)

        for _ in range(WARMUP):
            pg.allreduce_sum(sub_in, sub_out, n)
        pg.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(ITERS):
            pg.allreduce_sum(sub_in, sub_out, n)
        pg.synchronize()
        var us = Float64(perf_counter_ns() - t0) / 1e3 / Float64(ITERS)
        var bytes = n * 4
        var busbw = (
            2.0 * Float64(N - 1) / Float64(N) * Float64(bytes) / (us * 1e3)
        )

        # In place: output aliases input. Bypasses allreduce_sum's contract on
        # purpose — this is the M0 question.
        _fill(pg, sub_in, n)
        pg._allreduce_comm(sub_in, sub_in, n)
        pg.synchronize()
        var ok_inplace = _check(pg, sub_in, n)

        print(
            "  " + String(n), bytes, ok, Int(us),
            Float64(Int(busbw * 10)) / 10.0, ok_inplace, sep="\t",
        )
