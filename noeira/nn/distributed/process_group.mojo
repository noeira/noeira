"""ProcessGroup — N ranks driven from ONE host thread, and their collectives.

MAX's `comm` collectives are built for one process that drives every GPU: each
collective is N per-device calls, all enqueued before any can finish (the
kernels meet on `Signal` counters). `ProcessGroup` owns what those calls need
and the caller would otherwise build by hand — the N `DeviceContext`s, peer
access, the per-rank Signal buffers — and exposes arena-sized collectives over
flat fp32 `DeviceBuffer`s.

Two kinds of group:

- `ProcessGroup[N].devices(max_elems)` — rank r on GPU r, collectives from
  MAX's `comm` package. `backend` is `BACKEND_P2P` when every pair of GPUs
  has peer access, else `BACKEND_NAIVE` (MAX's host-staged allreduce).
- `ProcessGroup[N].shared(ctx)` — every rank on ONE context, collectives as
  local kernels. It is a simulator: no parallelism, but the same data flow,
  so the data-parallel correctness gates run on a Mac or a single GPU before
  any multi-GPU box is rented. Its allreduce reduces in rank order 0..N-1 into
  one buffer and copies it out, so every rank receives the same bits.

The allreduce here is a SUM. The 1/N of a mean is left to the caller, which
folds it into the copy back into the gradient arena (`DataParallel`).
"""

from std.sys import size_of, has_nvidia_gpu_accelerator
from std.collections import Array

from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.host._nvidia_cuda import CUDA

from layout import TileTensor, row_major
from comm import Signal, MAX_GPUS
from comm.sync import enable_p2p, init_signal_buffer
from comm.allreduce import allreduce
from comm.reducescatter import reducescatter
from comm.allgather import allgather

from noeira.nn.constants import DT, TPB


comptime BACKEND_SHARED = 0
"""Every rank on one context; collectives are local kernels (simulator)."""
comptime BACKEND_P2P = 1
"""Rank r on GPU r; MAX `comm` collectives over peer access."""
comptime BACKEND_NAIVE = 2
"""Rank r on GPU r; MAX `comm` host-staged fallback (no peer access)."""


def backend_name(b: Int) -> String:
    if b == BACKEND_SHARED:
        return "shared"
    if b == BACKEND_P2P:
        return "p2p"
    return "naive"


# ── shard partition ──────────────────────────────────────────────────────────


def shard_rows(rows: Int, nranks: Int, r: Int) -> Tuple[Int, Int]:
    """Rank r's `(first_row, n_rows)` when `rows` rows are split over `nranks`.

    ⚠ This is MAX's `ReduceScatterConfig` partition (`rank_unit_start` /
    `rank_units`), transcribed: rows split evenly, the first `rows % nranks`
    ranks take one more. The MAX backend does not take a partition as input, it
    COMPUTES this one, so the simulator and the sharded optimizer must use the
    same formula or rank r's optimizer would update rows another rank reduced.
    """
    var part = rows // nranks
    var rem = rows % nranks
    return (r * part + min(r, rem), part + (1 if r < rem else 0))


# ── local kernels (shared backend, and the scaled copy-back) ─────────────────


def _accum_kernel(
    acc: Pointer[Scalar[DT], MutAnyOrigin],
    src: Pointer[Scalar[DT], MutAnyOrigin],
    n_arg: Int64,
):
    """acc[i] += src[i]."""
    var n = Int(n_arg)
    var i = Int(global_idx.x)
    if i < n:
        acc[unsafe_offset=i] = acc[unsafe_offset=i] + src[unsafe_offset=i]


def _scale_copy_kernel(
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    src: Pointer[Scalar[DT], MutAnyOrigin],
    s: Scalar[DT],
    n_arg: Int64,
):
    """dst[i] = src[i] * s."""
    var n = Int(n_arg)
    var i = Int(global_idx.x)
    if i < n:
        dst[unsafe_offset=i] = src[unsafe_offset=i] * s


def scale_copy(
    dst: DeviceBuffer[DT],
    src: DeviceBuffer[DT],
    s: Scalar[DT],
    n: Int,
    ctx: DeviceContext,
) raises:
    """`dst = src * s` over `n` elements, enqueued on `ctx`."""
    if n == 0:
        return
    ctx.enqueue_function[_scale_copy_kernel](
        dst, src, s, Int64(n), grid_dim=(n + TPB - 1) // TPB, block_dim=TPB
    )


def _accum(
    acc: DeviceBuffer[DT], src: DeviceBuffer[DT], n: Int, ctx: DeviceContext
) raises:
    ctx.enqueue_function[_accum_kernel](
        acc, src, Int64(n), grid_dim=(n + TPB - 1) // TPB, block_dim=TPB
    )


def view_on(
    c: DeviceContext, buf: DeviceBuffer[DT], n: Int
) raises -> DeviceBuffer[DT]:
    """A non-owning view of `buf`'s first `n` elements, bound to `c` (same
    device). How MAX's own naive allreduce hands one context's buffers to
    another (`_allreduce_naive_single`)."""
    return DeviceBuffer[DT](
        c,
        rebind[MutPointer[Scalar[DT], MutAnyOrigin]](buf.unsafe_ptr()),
        n,
        owning=False,
    )


# ── the group ────────────────────────────────────────────────────────────────


struct ProcessGroup[N: Int](Movable):
    var ctxs: List[DeviceContext]
    """One per rank. In the shared backend, N handles on the same context."""
    var backend: Int
    var max_elems: Int
    """Largest collective (in fp32 elements) the Signal payloads are sized for."""
    var _sig_bufs: List[DeviceBuffer[DType.uint8]]
    var comm_ctxs: List[DeviceContext]
    """A SECOND context per rank, on the same GPU, for collectives that
    overlap the backward (`enable_comm_streams`). Empty until enabled. A
    `DeviceContext` is one stream, so a second context on a device is a
    second stream there; MAX's collectives take a context, not a stream."""

    def __init__(
        out self,
        var ctxs: List[DeviceContext],
        backend: Int,
        max_elems: Int,
        var sig_bufs: List[DeviceBuffer[DType.uint8]],
    ):
        self.ctxs = ctxs^
        self.backend = backend
        self.max_elems = max_elems
        self._sig_bufs = sig_bufs^
        self.comm_ctxs = List[DeviceContext]()

    @staticmethod
    def shared(ctx: DeviceContext) -> Self:
        """All N ranks on `ctx` (the simulator backend)."""
        var ctxs = List[DeviceContext](capacity=Self.N)
        for _ in range(Self.N):
            ctxs.append(ctx)
        return Self(
            ctxs^, BACKEND_SHARED, 0, List[DeviceBuffer[DType.uint8]]()
        )

    @staticmethod
    def devices(max_elems: Int) raises -> Self:
        """Rank r on GPU r, r in [0, N). Enables peer access when every pair
        supports it and sizes each rank's Signal buffer for collectives of up
        to `max_elems` fp32 elements.

        ⚠ The Signal payload is the staging area of the 2-stage allreduce, so
        it costs `size_of[Signal] + max_elems * 4` bytes PER GPU — for a
        gradient arena, one more arena-sized buffer. `comm`'s docstring asks
        for "the input tensor bytecount"; its own test allocates N times that.
        We follow the docstring (M0 checks it)."""
        comptime assert Self.N >= 2, "ProcessGroup.devices needs N >= 2"
        comptime assert Self.N <= MAX_GPUS, "MAX comm supports at most 8 GPUs"
        if DeviceContext.number_of_devices() < Self.N:
            raise Error(
                "ProcessGroup.devices: "
                + String(Self.N)
                + " ranks but "
                + String(DeviceContext.number_of_devices())
                + " GPU(s)"
            )
        var ctxs = List[DeviceContext](capacity=Self.N)
        for r in range(Self.N):
            ctxs.append(DeviceContext(device_id=r))
        var p2p = enable_p2p()
        var sig_bytes = size_of[Signal]() + max_elems * size_of[Scalar[DT]]()
        var sigs = List[DeviceBuffer[DType.uint8]](capacity=Self.N)
        for r in range(Self.N):
            sigs.append(ctxs[r].create_buffer_sync[DType.uint8](sig_bytes))
        for r in range(Self.N):
            init_signal_buffer(sigs[r], ctxs[r])
        for r in range(Self.N):
            ctxs[r].synchronize()
        return Self(
            ctxs^, BACKEND_P2P if p2p else BACKEND_NAIVE, max_elems, sigs^
        )

    def ctx(self, r: Int) -> DeviceContext:
        return self.ctxs[r]

    def synchronize(self) raises:
        """Wait for every rank's stream (and its comm stream, if any)."""
        for r in range(Self.N):
            self.ctxs[r].synchronize()
        for r in range(len(self.comm_ctxs)):
            self.comm_ctxs[r].synchronize()

    def enable_comm_streams(mut self) raises:
        """Create one comm context per rank on the rank's GPU (idempotent).

        Shared backend: ONE extra context, shared by every rank like the main
        one. On NVIDIA, prints a warning if MAX hands back the main stream
        (then nothing overlaps, but every result is unchanged)."""
        if len(self.comm_ctxs) > 0:
            return
        if self.backend == BACKEND_SHARED:
            var c = DeviceContext(device_id=Int(self.ctxs[0].id()))
            for _ in range(Self.N):
                self.comm_ctxs.append(c)
        else:
            for r in range(Self.N):
                self.comm_ctxs.append(DeviceContext(device_id=r))
        comptime if has_nvidia_gpu_accelerator():
            var a = CUDA(self.ctxs[0].stream())
            var b = CUDA(self.comm_ctxs[0].stream())
            var pa = Int(a.value()) if a else 0
            var pb = Int(b.value()) if b else 0
            if pa == pb:
                print(
                    "[ProcessGroup] WARNING: the comm context runs on the"
                    " main stream; collectives will not overlap compute"
                )

    def comm_ctx(self, r: Int) -> DeviceContext:
        return self.comm_ctxs[r]

    def has_comm_streams(self) -> Bool:
        return len(self.comm_ctxs) > 0

    def graph_ctxs(self) -> List[DeviceContext]:
        """The distinct contexts, one graph each (`RankGraphs`): every rank's
        on the devices, the one shared context on the simulator."""
        var out = List[DeviceContext]()
        if self.backend == BACKEND_SHARED:
            out.append(self.ctxs[0])
        else:
            for r in range(Self.N):
                out.append(self.ctxs[r])
        return out^

    def capturable_collectives(self) -> Bool:
        """Whether a collective may sit inside a captured graph. The NAIVE
        (host-staged) allreduce allocates its staging buffers per call, which
        aborts a capture; the P2P and simulator paths allocate nothing."""
        return self.backend != BACKEND_NAIVE

    def _stream_barrier(self) raises:
        """Every rank's stream waits for everything already enqueued on every
        other rank's stream (device-side events, no host sync).

        Needed around MAX's NAIVE collectives only. There rank r's stream
        copies peer k's input buffer (`enqueue_copy(scratch, in_k)`,
        `_allreduce_naive_single`) with no ordering against rank k's stream:
        it can read `in_k` before rank k's backward finished writing it, and
        rank k can overwrite `in_k` (the 1/N scale, the next `zero_grad`)
        before rank r copied it. The P2P kernels order both through their
        Signal barriers. Without this the GPT gate drifted (replicas 2.2e-2
        apart after 30 steps, H200 NVL, PCIe without native atomics).
        Called before a collective (its inputs are complete) and after it
        (no rank moves on while a peer still reads its buffers)."""
        for r in range(Self.N):
            for k in range(Self.N):
                if k != r:
                    self.ctxs[r].enqueue_wait_for(self.ctxs[k])

    # ── collectives ──────────────────────────────────────────────────────────

    def allreduce_sum(
        mut self,
        ins: List[DeviceBuffer[DT]],
        outs: List[DeviceBuffer[DT]],
        n: Int,
    ) raises:
        """`outs[r] = sum_k ins[k]` over the first `n` elements, every rank.

        Enqueues rank by rank from this thread and does not block it. Inputs
        must not alias outputs (MAX's 1-stage kernel writes `out_r` while
        peers still read `in_r`)."""
        if Self.N == 1:
            if n > 0:
                self.ctxs[0].enqueue_copy(
                    outs[0].create_sub_buffer[DT](0, n),
                    ins[0].create_sub_buffer[DT](0, n),
                )
            return
        if self.backend == BACKEND_SHARED:
            var c = self.ctxs[0]
            self._allreduce_shared(ins, outs, n, c)
            return
        self._check_payload(n)
        if self.backend == BACKEND_NAIVE:
            self._stream_barrier()
        self._allreduce_comm[False](ins, outs, n)
        if self.backend == BACKEND_NAIVE:
            self._stream_barrier()

    def allreduce_sum_comm(
        mut self,
        ins: List[DeviceBuffer[DT]],
        outs: List[DeviceBuffer[DT]],
        n: Int,
    ) raises:
        """`allreduce_sum` enqueued on the COMM contexts, with NO ordering of
        its own: the caller makes each comm stream wait for the inputs (every
        rank's, on the naive and shared backends, whose ranks read their
        peers' buffers) and makes the main streams wait for it before they
        touch `ins` or `outs` again (`DataParallel` with overlap)."""
        if Self.N == 1:
            if n > 0:
                var c = self.comm_ctxs[0]
                c.enqueue_copy(view_on(c, outs[0], n), view_on(c, ins[0], n))
            return
        if self.backend == BACKEND_SHARED:
            # MAX refuses a copy whose destination was allocated by another
            # context ("device context does not match context of dst"), even
            # on the same device: hand the comm context views of the buffers.
            var c = self.comm_ctxs[0]
            var vi = List[DeviceBuffer[DT]](capacity=Self.N)
            var vo = List[DeviceBuffer[DT]](capacity=Self.N)
            for r in range(Self.N):
                vi.append(view_on(c, ins[r], n))
                vo.append(view_on(c, outs[r], n))
            self._allreduce_shared(vi, vo, n, c)
            return
        self._check_payload(n)
        self._allreduce_comm[True](ins, outs, n)

    def _check_payload(self, n: Int) raises:
        if n > self.max_elems:
            raise Error(
                "allreduce_sum: "
                + String(n)
                + " elements but the Signal payloads were sized for "
                + String(self.max_elems)
            )

    def _allreduce_shared(
        mut self,
        ins: List[DeviceBuffer[DT]],
        outs: List[DeviceBuffer[DT]],
        n: Int,
        c: DeviceContext,
    ) raises:
        if n == 0:
            return
        var acc = outs[0].create_sub_buffer[DT](0, n)
        c.enqueue_copy(acc, ins[0].create_sub_buffer[DT](0, n))
        for k in range(1, Self.N):
            _accum(acc, ins[k], n, c)
        for r in range(1, Self.N):
            c.enqueue_copy(outs[r].create_sub_buffer[DT](0, n), acc)

    def _allreduce_comm[
        ON_COMM: Bool
    ](
        mut self,
        ins: List[DeviceBuffer[DT]],
        outs: List[DeviceBuffer[DT]],
        n: Int,
    ) raises:
        """MAX `comm` allreduce on the main contexts, or on the comm ones."""
        comptime if has_nvidia_gpu_accelerator() and Self.N >= 2:
            comptime InT = TileTensor[
                DT, type_of(row_major(0)), ImmutAnyOrigin
            ]
            var in_t = Array[InT, Self.N](uninitialized=True)
            for k in range(Self.N):
                in_t[k] = TileTensor(
                    rebind[ImmPointer[Scalar[DT], ImmutAnyOrigin]](
                        ins[k].unsafe_ptr()
                    ),
                    row_major(n),
                )
            var sigs = self._rank_sigs()
            comptime for r in range(Self.N):
                var out_t = TileTensor(
                    rebind[MutPointer[Scalar[DT], MutAnyOrigin]](
                        outs[r].unsafe_ptr()
                    ),
                    row_major(n),
                )
                comptime if ON_COMM:
                    allreduce[ngpus=Self.N](in_t, out_t, sigs, self.comm_ctxs[r])
                else:
                    allreduce[ngpus=Self.N](in_t, out_t, sigs, self.ctxs[r])
        else:
            raise Error("allreduce_sum: MAX comm collectives need NVIDIA GPUs")

    def reduce_scatter_sum(
        mut self,
        ins: List[DeviceBuffer[DT]],
        outs: List[DeviceBuffer[DT]],
        rows: Int,
        unit: Int,
    ) raises:
        """`outs[r] = sum_k ins[k][rows of r]`, every rank.

        The inputs are viewed as `[rows, unit]`; rank r receives its
        `shard_rows(rows, N, r)` rows (`n_rows * unit` elements in `outs[r]`).
        MAX's reduce-scatter has NO non-P2P path: on `BACKEND_NAIVE` this
        raises, and the caller falls back to an allreduce."""
        if Self.N == 1:
            if rows > 0:
                self.ctxs[0].enqueue_copy(
                    outs[0].create_sub_buffer[DT](0, rows * unit),
                    ins[0].create_sub_buffer[DT](0, rows * unit),
                )
            return
        if self.backend == BACKEND_SHARED:
            var c = self.ctxs[0]
            for r in range(Self.N):
                var sh = shard_rows(rows, Self.N, r)
                var off = sh[0] * unit
                var n = sh[1] * unit
                if n == 0:
                    continue
                var acc = outs[r].create_sub_buffer[DT](0, n)
                # Rank order 0..N-1, as `_allreduce_shared`: the shard of the
                # sum is bit-identical to the same rows of the allreduce.
                c.enqueue_copy(acc, ins[0].create_sub_buffer[DT](off, n))
                for k in range(1, Self.N):
                    _accum(acc, ins[k].create_sub_buffer[DT](off, n), n, c)
            return
        if self.backend != BACKEND_P2P:
            raise Error(
                "reduce_scatter_sum: MAX's reduce-scatter requires P2P access"
            )
        comptime if has_nvidia_gpu_accelerator() and Self.N >= 2:
            comptime InT = TileTensor[
                DT, type_of(row_major(0, 0)), ImmutAnyOrigin
            ]
            comptime OutT = TileTensor[
                DT, type_of(row_major(0, 0)), MutAnyOrigin
            ]
            var in_t = Array[InT, Self.N](uninitialized=True)
            var out_t = Array[OutT, Self.N](uninitialized=True)
            for k in range(Self.N):
                in_t[k] = TileTensor(
                    rebind[ImmPointer[Scalar[DT], ImmutAnyOrigin]](
                        ins[k].unsafe_ptr()
                    ),
                    row_major(rows, unit),
                )
                var sh = shard_rows(rows, Self.N, k)
                out_t[k] = TileTensor(
                    rebind[MutPointer[Scalar[DT], MutAnyOrigin]](
                        outs[k].unsafe_ptr()
                    ),
                    row_major(sh[1], unit),
                )
            var sigs = self._rank_sigs()
            comptime for r in range(Self.N):
                reducescatter[ngpus=Self.N](
                    in_t, out_t, sigs, self.ctxs[r], my_rank=r
                )
        else:
            raise Error("reduce_scatter_sum: MAX comm needs NVIDIA GPUs")

    def all_gather_rows(
        mut self, bufs: List[DeviceBuffer[DT]], rows: Int, unit: Int
    ) raises:
        """In place over full-length buffers viewed as `[rows, unit]`: every
        rank's own `shard_rows` rows are copied into the same rows of every
        other rank's buffer.

        On the MAX backend the outputs are sub-buffers of the destination
        buffers themselves (`allgather` writes one output per source rank), so
        nothing is staged. ⚠ The self slot (rank r's own shard into rank r)
        aliases input and output: each element is read and rewritten with its
        own value — M3 checks MAX tolerates that on the box."""
        if Self.N == 1:
            return
        if self.backend == BACKEND_SHARED:
            var c = self.ctxs[0]
            for k in range(Self.N):
                var sh = shard_rows(rows, Self.N, k)
                var off = sh[0] * unit
                var n = sh[1] * unit
                if n == 0:
                    continue
                var src = bufs[k].create_sub_buffer[DT](off, n)
                for r in range(Self.N):
                    if r != k:
                        c.enqueue_copy(
                            bufs[r].create_sub_buffer[DT](off, n), src
                        )
            return
        comptime if has_nvidia_gpu_accelerator() and Self.N >= 2:
            comptime T1 = TileTensor[DT, type_of(row_major(0)), MutAnyOrigin]
            comptime InT = TileTensor[
                DT, type_of(row_major(0)), ImmutAnyOrigin
            ]
            var in_t = Array[InT, Self.N](uninitialized=True)
            var out_t = Array[T1, Self.N * Self.N](uninitialized=True)
            for k in range(Self.N):
                var sh = shard_rows(rows, Self.N, k)
                var off = sh[0] * unit
                var n = sh[1] * unit
                in_t[k] = TileTensor(
                    rebind[ImmPointer[Scalar[DT], ImmutAnyOrigin]](
                        bufs[k].unsafe_ptr().unsafe_offset(off)
                    ),
                    row_major(n),
                )
                for r in range(Self.N):
                    out_t[r * Self.N + k] = TileTensor(
                        rebind[MutPointer[Scalar[DT], MutAnyOrigin]](
                            bufs[r].unsafe_ptr().unsafe_offset(off)
                        ),
                        row_major(n),
                    )
            var sigs = self._rank_sigs()
            if self.backend == BACKEND_NAIVE:
                self._stream_barrier()
            comptime for r in range(Self.N):
                allgather[ngpus=Self.N](
                    in_t, out_t, sigs, self.ctxs[r], my_rank=r
                )
            if self.backend == BACKEND_NAIVE:
                self._stream_barrier()
        else:
            raise Error("all_gather_rows: MAX comm needs NVIDIA GPUs")

    def all_gather_into(
        mut self,
        srcs: List[DeviceBuffer[DT]],
        dsts: List[DeviceBuffer[DT]],
        rows: Int,
        unit: Int,
    ) raises:
        """Out of place: rank k's `shard_rows(rows, N, k)` rows, held
        contiguously at the start of `srcs[k]`, land at the same rows of every
        rank's `dsts[r]` (viewed as `[rows, unit]`). ZeRO-3's per-unit weight
        gather (shard -> full unit) and ZeRO-2's refresh of the replicas.

        The caller keeps the sources unwritten until every rank has read them
        and does not read the destinations before the gather ends; the P2P
        kernels barrier at both ends, and the naive path is bracketed by
        `_stream_barrier` here."""
        if Self.N == 1:
            if rows > 0:
                self.ctxs[0].enqueue_copy(
                    dsts[0].create_sub_buffer[DT](0, rows * unit),
                    srcs[0].create_sub_buffer[DT](0, rows * unit),
                )
            return
        if self.backend == BACKEND_SHARED:
            var c = self.ctxs[0]
            for k in range(Self.N):
                var sh = shard_rows(rows, Self.N, k)
                var off = sh[0] * unit
                var n = sh[1] * unit
                if n == 0:
                    continue
                var src = srcs[k].create_sub_buffer[DT](0, n)
                for r in range(Self.N):
                    c.enqueue_copy(dsts[r].create_sub_buffer[DT](off, n), src)
            return
        comptime if has_nvidia_gpu_accelerator() and Self.N >= 2:
            comptime T1 = TileTensor[DT, type_of(row_major(0)), MutAnyOrigin]
            comptime InT = TileTensor[
                DT, type_of(row_major(0)), ImmutAnyOrigin
            ]
            var in_t = Array[InT, Self.N](uninitialized=True)
            var out_t = Array[T1, Self.N * Self.N](uninitialized=True)
            for k in range(Self.N):
                var sh = shard_rows(rows, Self.N, k)
                var off = sh[0] * unit
                var n = sh[1] * unit
                in_t[k] = TileTensor(
                    rebind[ImmPointer[Scalar[DT], ImmutAnyOrigin]](
                        srcs[k].unsafe_ptr()
                    ),
                    row_major(n),
                )
                for r in range(Self.N):
                    out_t[r * Self.N + k] = TileTensor(
                        rebind[MutPointer[Scalar[DT], MutAnyOrigin]](
                            dsts[r].unsafe_ptr().unsafe_offset(off)
                        ),
                        row_major(n),
                    )
            var sigs = self._rank_sigs()
            if self.backend == BACKEND_NAIVE:
                self._stream_barrier()
            comptime for r in range(Self.N):
                allgather[ngpus=Self.N](
                    in_t, out_t, sigs, self.ctxs[r], my_rank=r
                )
            if self.backend == BACKEND_NAIVE:
                self._stream_barrier()
        else:
            raise Error("all_gather_into: MAX comm needs NVIDIA GPUs")

    def stream_barrier(self) raises:
        """Every rank's stream waits for every peer's (device-side). For a
        driver that orders its own reads of peers' buffers; no-op on the
        simulator, whose ranks share one stream."""
        if Self.N == 1 or self.backend == BACKEND_SHARED:
            return
        self._stream_barrier()

    def _rank_sigs(
        mut self,
    ) -> Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS]:
        var sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
            uninitialized=True
        )
        for k in range(len(self._sig_bufs)):
            sigs[k] = (
                self._sig_bufs[k]
                .unsafe_ptr()
                .unsafe_bitcast[Signal]()
                .as_unsafe_any_origin()
            )
        return sigs^

    def broadcast(
        mut self, root: Int, bufs: List[DeviceBuffer[DT]], n: Int
    ) raises:
        """`bufs[r] = bufs[root]` for every rank, over `n` elements.

        A setup-time operation, not a hot-path one: it synchronizes the root
        before the copies and every rank after. The copies are plain
        `enqueue_copy`s, device to device (through the host without peer
        access), which is enough for the initial weight broadcast."""
        if Self.N == 1 or n == 0:
            return
        self.ctxs[root].synchronize()
        var src = bufs[root].create_sub_buffer[DT](0, n)
        for r in range(Self.N):
            if r != root:
                self.ctxs[r].enqueue_copy(
                    bufs[r].create_sub_buffer[DT](0, n), src
                )
        self.synchronize()
