"""Split test: is it the CAPTURE, or what we put IN it?

`cuStreamEndCapture` faults inside libcuda on NVIDIA (MAX 26.5.0rc2), and so
does `cuStreamGetCaptureInfo_v2`. Two different driver entry points crashing
while merely READING the capture state says the capture is already corrupt
when we touch it — but not WHY. Two stories remain, and they lead opposite
ways:

  A. The begin/end pair on MAX's borrowed stream is broken by itself.
     Then nothing inside the window matters, MAX's mid-capture
     `cuStreamSynchronize` is a red herring, and stream-borrowing is simply
     dead under this MAX — stop repairing it and turn graphs off.

  B. The window's CONTENTS corrupt it — the recorded kernel, or the
     synchronize we suppress mid-capture. Then capture machinery is fine and
     there is something specific left to fix.

This probe is story A's test, and it is decisive because it removes
EVERYTHING from the window: no kernel, no enqueue, hence no MAX call at all
between begin and end.

    faults    -> A. begin/end alone is broken. Stop.
    survives  -> B. Capture works empty; the fault comes from the contents.
                 `end_capture` will then raise "Captured 0 nodes", which is
                 the EXPECTED, CORRECT outcome here — an empty capture really
                 does hold zero nodes. Reaching that error is a PASS.

⚠ HISTORICAL NOTE — IT WAS NEITHER STORY. This probe faulted, and that was
read as story A ("capture itself is broken"), which sent the investigation
toward disabling CUDA graphs entirely. The real cause was outside both
branches: `ctx`'s last use was `CUDAGraph(ctx)`, so Mojo destroyed the
DeviceContext — and MAX's destructor destroyed the stream — before
`begin_capture` ever ran. `CUDAGraph` now HOLDS the context, so the premise
this split depends on (that the window is the only variable) finally holds.
Kept as a lesson: a two-way split is only decisive if the answer is actually
one of the two.

⚠ THE ZERO-NODE ERROR IS THE SUCCESS CASE. Do not "fix" it. The question is
only whether we get there at all, or die inside the driver first.

⚠ Run on a TTY. Mojo's stdout is block-buffered and an abort discards it, so
the only output that reliably survives is the shim's `[intercept]` stderr —
which is precisely why the interceptor prints the begin/end rc values.

    pixi run -e nvidia mojo run -I . tests/cuda/probe_empty_capture.mojo
"""

from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from std.sys import default_accelerator
from noeira.core.tiles import Tile, tile

from noeira.nn.constants import DT
from noeira.cuda import CUDAGraph


def main() raises:
    comptime if not default_accelerator().is_nvidia_gpu():
        print("non-NVIDIA: capture is a comptime no-op here. Skipped.")
        return

    @always_inline
    def _bump(buf: Tile[DT, 1]):
        if Int(thread_idx.x) != 0:
            return
        buf[0] = buf[0] + Scalar[DT](1.0)

    var ctx = DeviceContext()
    var buf = ctx.enqueue_create_buffer[DT](1)
    var lt = tile[1](buf)

    # Stream discovery still needs one launch — but it happens BEFORE the
    # capture window opens, so it is not part of what we are testing.
    print("[probe] warmup launch (outside the window) ...")
    ctx.enqueue_function[_bump](lt, grid_dim=1, block_dim=1)
    ctx.synchronize()

    var g = CUDAGraph(ctx)

    print("[probe] begin_capture ...")
    g.begin_capture()

    # ⚠ DELIBERATELY EMPTY. Not an oversight — the whole point is that no
    # kernel is enqueued and MAX is never called between begin and end, so
    # neither the recorded launch nor MAX's mid-capture synchronize can be
    # implicated in whatever happens next.
    print("[probe] window is empty by design; end_capture ...")

    try:
        g.end_capture()
        print("[probe] end_capture returned. nodes =", g.num_nodes())
    except e:
        # "Captured 0 nodes" is the expected, correct error for an empty
        # capture and means story B: the machinery works.
        print("[probe] end_capture raised:", e)
        print("[probe] SURVIVED — the begin/end pair itself is sound, so the")
        print("        fault comes from the window's CONTENTS (story B).")
        return

    print("[probe] SURVIVED the empty capture (story B).")
