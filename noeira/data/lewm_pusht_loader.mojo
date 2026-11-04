"""LewmBatchLoader — PushT training batches read on their own thread.

docs/LEWM_REOPEN_PLAN.md P6.2. One training step at batch 128 enqueues ~4,000
kernels; once the driver's launch queue is full each launch waits for the GPU,
so the main thread is held for most of the step and cannot also read the next
batch (0.19 s of HDF5 per 128 windows). This worker does the reading:

    var loader = LewmBatchLoaderThread[B, T, IMG, ACT_IN](
        h5, clip_idx, a_mean, a_std, [tr.staging(0), tr.staging(1)])
    loader.request(0)
    for s in range(n):
        loader.wait(s)                       # batch s is in slot s % 2
        if s + 1 < n: loader.request(s + 1)  # BEFORE the submit: it blocks
        tr.submit_staged(loader.actions(s), s % 2)
        var st = tr.finish()

Batch `k` is the windows `clip_idx[k*B .. (k+1)*B)`: the frames go straight
into the trainer's pinned staging slot `k % 2`, the actions — z-scored per raw
dimension with the given mean / std, NaN (episode ends) to 0, exactly
`train.py`'s normaliser + `nan_to_num` — into a raw two-slot buffer.

⚠ Protocol: request `k + 1` only after `finish()` of step `k - 1` (the slot it
overwrites was read by that step's async upload). The loop above does — and
requests BEFORE submitting step `k`, whose launches hold the main thread for
most of the step (requested after, the read started 0.37 s late and the step
waited 0.16 s for it).

The window shape is the model's (`noeira/deep_agents/lewm/trainer.mojo`:
`T` frames of `IMG`² pixels, `ACT_IN` = frameskip x 2 raw action words per
frame); the loader reads it from the dataset with frameskip `ACT_IN / 2`.

Threading: the worker opens its OWN `LewmPushTExpert` in `on_start` (libhdf5
is touched by one thread only); everything the main thread reads back crosses
through raw memory and the `SharedBlock` cells (release store / acquire load) —
never a Mojo `List` (`core/concurrent/worker.mojo`).
"""

from std.memory import alloc
from std.memory.alloc import Layout as AllocLayout
from std.math import isnan
from std.time import perf_counter_ns

from noeira.nn.constants import DT
from noeira.nn.datasets.lewm_pusht import LewmPushTExpert
from noeira.core.concurrent.worker import (
    BackgroundWorker, BackgroundThread, WorkerCtl, POLL_DID_WORK, POLL_IDLE,
)
from noeira.core.concurrent.block import SharedBlock
from noeira.core.concurrent.thread import sleep_us

comptime CELL_REQ = 0
"""The batch index the main thread wants next (-1: none)."""
comptime CELL_DONE = 1
"""The last batch index the worker finished."""
comptime CELL_ERR = 2
"""1 if the worker failed (open or read); it then stops serving."""
comptime CELL_LOAD_NS = 3
"""Nanoseconds the last batch took to read (diagnostics)."""


struct LewmBatchLoader[B: Int, T: Int, IMG: Int, ACT_IN: Int](BackgroundWorker):
    comptime _HWC = Self.IMG * Self.IMG * 3
    comptime _ACT = Self.T * Self.ACT_IN

    var h5: String
    var clip_idx: List[Int]
    var mean: Float32
    var mean2: Float32
    var std: Float32
    var std2: Float32
    var stage_addr: List[Int]
    var act_addr: Int
    var cells: SharedBlock
    var ds: Optional[LewmPushTExpert]
    var done: Int

    def __init__(
        out self, h5: String, clip_idx: List[Int], a_mean: List[Float32],
        a_std: List[Float32], stage_addr: List[Int], act_addr: Int,
        cells: SharedBlock,
    ):
        self.h5 = h5
        self.clip_idx = clip_idx.copy()
        self.mean = a_mean[0]
        self.mean2 = a_mean[1]
        self.std = a_std[0]
        self.std2 = a_std[1]
        self.stage_addr = stage_addr.copy()
        self.act_addr = act_addr
        self.cells = cells
        self.ds = None
        self.done = -1

    def __init__(out self, *, deinit move: Self):
        self.h5 = move.h5^
        self.clip_idx = move.clip_idx^
        self.mean = move.mean
        self.mean2 = move.mean2
        self.std = move.std
        self.std2 = move.std2
        self.stage_addr = move.stage_addr^
        self.act_addr = move.act_addr
        self.cells = move.cells^
        self.ds = move.ds^
        self.done = move.done

    def on_start(mut self, ctl: WorkerCtl):
        try:
            self.ds = LewmPushTExpert(
                frameskip=Self.ACT_IN // 2, num_steps=Self.T, path=self.h5
            )
        except:
            self.cells.release_store(CELL_ERR, 1)

    def poll(mut self, ctl: WorkerCtl) -> Int:
        if self.cells.acquire_load(CELL_ERR) != 0:
            return POLL_IDLE
        var req = Int(self.cells.acquire_load(CELL_REQ))
        if req <= self.done:
            return POLL_IDLE
        var t0 = perf_counter_ns()
        try:
            self._load(req)
        except:
            self.cells.release_store(CELL_ERR, 1)
            return POLL_IDLE
        self.done = req
        self.cells.relaxed_store(CELL_LOAD_NS, Int64(perf_counter_ns() - t0))
        self.cells.release_store(CELL_DONE, Int64(req))
        return POLL_DID_WORK

    def on_stop(mut self, ctl: WorkerCtl):
        self.ds = None

    def _load(mut self, k: Int) raises:
        var slot = k % 2
        var stage = Pointer[Scalar[DType.uint8], MutAnyOrigin](
            unsafe_from_address=self.stage_addr[slot]
        )
        var act = Pointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=self.act_addr
        ).unsafe_offset(slot * Self.B * Self._ACT)
        var araw = List[Float32](length=Self._ACT, fill=0)
        var dense = List[UInt8](length=Self.T * (Self.ACT_IN // 2) * Self._HWC, fill=0)
        for b in range(Self.B):
            self.ds.value().sample_clip_pixels_uint8(
                self.clip_idx[k * Self.B + b],
                stage.unsafe_offset(b * Self.T * Self._HWC),
                rebind[Pointer[Scalar[DType.float32], MutAnyOrigin]](araw.unsafe_ptr()),
                rebind[Pointer[Scalar[DType.uint8], MutAnyOrigin]](dense.unsafe_ptr()),
            )
            for j in range(Self._ACT):
                var m = self.mean if j % 2 == 0 else self.mean2
                var sd = self.std if j % 2 == 0 else self.std2
                var z = (araw[j] - m) / sd
                act[unsafe_offset=b * Self._ACT + j] = Scalar[DT](0.0) if isnan(z) else rebind[Scalar[DT]](z)


struct LewmBatchLoaderThread[B: Int, T: Int, IMG: Int, ACT_IN: Int](Movable):
    """Owns the worker thread, the cells and the action buffer."""

    comptime _ACT = Self.T * Self.ACT_IN

    var thread: BackgroundThread[LewmBatchLoader[Self.B, Self.T, Self.IMG, Self.ACT_IN]]
    var cells: SharedBlock
    var act: Pointer[Scalar[DT], MutUntrackedOrigin]

    def __init__(
        out self, h5: String, clip_idx: List[Int], a_mean: List[Float32],
        a_std: List[Float32], stages: List[Pointer[Scalar[DType.uint8], MutAnyOrigin]],
    ) raises:
        comptime assert Self.ACT_IN % 2 == 0, "PushT actions are 2-D: ACT_IN = frameskip x 2"
        if len(clip_idx) % Self.B != 0:
            raise Error("LewmBatchLoaderThread: clip_idx is not a whole number of batches")
        self.cells = SharedBlock(8)
        self.cells.release_store(CELL_REQ, -1)
        self.cells.release_store(CELL_DONE, -1)
        self.act = alloc(AllocLayout[Scalar[DT]](count=2 * Self.B * Self._ACT)).unsafe_leak()
        var addrs = List[Int]()
        for st in stages:
            addrs.append(Int(st))
        self.thread = BackgroundThread(LewmBatchLoader[Self.B, Self.T, Self.IMG, Self.ACT_IN](
            h5, clip_idx, a_mean, a_std, addrs, Int(self.act), self.cells
        ))

    def request(mut self, k: Int):
        """Ask for batch `k` (see the protocol in the module header)."""
        self.cells.release_store(CELL_REQ, Int64(k))

    def wait(mut self, k: Int) raises -> Float64:
        """Block until batch `k` is in its slot; returns how long the worker
        spent reading it (seconds)."""
        while Int(self.cells.acquire_load(CELL_DONE)) < k:
            if self.cells.acquire_load(CELL_ERR) != 0:
                raise Error("LewmBatchLoader: the worker failed (open or read)")
            _ = sleep_us(100)
        return Float64(self.cells.relaxed_load(CELL_LOAD_NS)) / 1e9

    def actions(self, k: Int) -> List[Scalar[DT]]:
        """Batch `k`'s actions (B, T, ACT_IN), copied out of the shared slot."""
        var out = List[Scalar[DT]](capacity=Self.B * Self._ACT)
        var base = (k % 2) * Self.B * Self._ACT
        for i in range(Self.B * Self._ACT):
            out.append(self.act[unsafe_offset=base + i])
        return out^

    def stop(mut self) raises:
        """Join the worker, then free the action slots it wrote."""
        self.thread.stop(drain_ms=0)
        self.act.unsafe_free()
