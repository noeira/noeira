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

from layout import TileTensor, row_major
from comm import Signal, MAX_GPUS
from comm.sync import enable_p2p, init_signal_buffer
from comm.allreduce import allreduce

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


# ── the group ────────────────────────────────────────────────────────────────


struct ProcessGroup[N: Int](Movable):
    var ctxs: List[DeviceContext]
    """One per rank. In the shared backend, N handles on the same context."""
    var backend: Int
    var max_elems: Int
    """Largest collective (in fp32 elements) the Signal payloads are sized for."""
    var _sig_bufs: List[DeviceBuffer[DType.uint8]]

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
        """Wait for every rank's stream."""
        for r in range(Self.N):
            self.ctxs[r].synchronize()

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
            self._allreduce_shared(ins, outs, n)
            return
        if n > self.max_elems:
            raise Error(
                "allreduce_sum: "
                + String(n)
                + " elements but the Signal payloads were sized for "
                + String(self.max_elems)
            )
        self._allreduce_comm(ins, outs, n)

    def _allreduce_shared(
        mut self,
        ins: List[DeviceBuffer[DT]],
        outs: List[DeviceBuffer[DT]],
        n: Int,
    ) raises:
        if n == 0:
            return
        var c = self.ctxs[0]
        var acc = outs[0].create_sub_buffer[DT](0, n)
        c.enqueue_copy(acc, ins[0].create_sub_buffer[DT](0, n))
        for k in range(1, Self.N):
            _accum(acc, ins[k], n, c)
        for r in range(1, Self.N):
            c.enqueue_copy(outs[r].create_sub_buffer[DT](0, n), acc)

    def _allreduce_comm(
        mut self,
        ins: List[DeviceBuffer[DT]],
        outs: List[DeviceBuffer[DT]],
        n: Int,
    ) raises:
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
            var sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
                uninitialized=True
            )
            for k in range(Self.N):
                sigs[k] = (
                    self._sig_bufs[k]
                    .unsafe_ptr()
                    .unsafe_bitcast[Signal]()
                    .as_unsafe_any_origin()
                )
            comptime for r in range(Self.N):
                var out_t = TileTensor(
                    rebind[MutPointer[Scalar[DT], MutAnyOrigin]](
                        outs[r].unsafe_ptr()
                    ),
                    row_major(n),
                )
                allreduce[ngpus=Self.N](in_t, out_t, sigs, self.ctxs[r])
        else:
            raise Error("allreduce_sum: MAX comm collectives need NVIDIA GPUs")

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
