"""Does MAX destroy its stream on its own — with CUDAGraph nowhere in sight?

⚠ THIS EXISTS BECAUSE I ASSERTED SOMETHING I HAD NOT TESTED. I concluded
"MAX 26.5.0rc2 destroys its stream" from traces taken while `CUDAGraph` was
running, which cannot distinguish MAX's own behaviour from something OUR
code provokes — and the reasonable objection is that MAX destroying streams
per-synchronize would be strange, and that graph capture demonstrably worked
before, so the change is more likely ours.

This probe removes every one of our moving parts. It never constructs a
`CUDAGraph`, never dlopens the interceptor, never calls a single
`intercept_*` entry point. It only launches kernels and synchronizes. The
interceptor is present purely because LD_PRELOAD puts it there, and its
`cuStreamDestroy` hook is passive.

    cuStreamDestroy fires between the markers   -> MAX tears its stream down
                                                   on its own. Our Mojo file
                                                   is exonerated; the
                                                   borrowed-stream premise
                                                   really is gone.
    no destroy at all                           -> MAX keeps its stream, and
                                                   the destroys we saw were
                                                   PROVOKED by something
                                                   CUDAGraph does. The bug is
                                                   ours and this arc's
                                                   conclusion is wrong.

⚠ NEEDS NOEIRA_INTERCEPT_LOG=1. The destroy log is rate-limited to 2 lines
by default (it is on the hot path — MAX's teardown happens often enough that
logging it unconditionally slowed training measurably). The env var lifts the
limit AND is what makes the per-launch stream lines visible.

⚠ `flush=True` ON EVERY PRINT IS LOAD-BEARING. Mojo's stdout is block
buffered while the interceptor writes to unbuffered stderr, so without
flushing, the two streams interleave in the wrong order and the whole
question — WHERE does the destroy fall relative to our markers — becomes
unanswerable. This is the same buffering that made three crash runs of this
arc look like they died earlier than they did.

    NOEIRA_INTERCEPT_LOG=1 pixi run -e nvidia mojo run -I . \
        tests/cuda/probe_max_stream_lifetime.mojo
"""

from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from std.sys import default_accelerator
from noeira.core.tiles import Tile, tile

from noeira.nn.constants import DT


def main() raises:
    comptime if not default_accelerator().is_nvidia_gpu():
        print("non-NVIDIA: no interceptor, nothing to observe. Skipped.")
        return

    @always_inline
    def _bump(buf: Tile[DT, 1]):
        if Int(thread_idx.x) != 0:
            return
        buf[0] = buf[0] + Scalar[DT](1.0)

    var ctx = DeviceContext()
    var buf = ctx.enqueue_create_buffer[DT](1)
    var lt = tile[1](buf)

    print("[probe] === launch 1 ===", flush=True)
    ctx.enqueue_function[_bump](lt, grid_dim=1, block_dim=1)
    print("[probe] launched; about to synchronize", flush=True)
    ctx.synchronize()
    print("[probe] synchronize #1 returned", flush=True)

    # If a destroy appears above this line, MAX tears the stream down as part
    # of synchronize — with no CUDAGraph in the process at all.
    print("[probe] === launch 2 ===", flush=True)
    ctx.enqueue_function[_bump](lt, grid_dim=1, block_dim=1)
    print("[probe] launched; about to synchronize", flush=True)
    ctx.synchronize()
    print("[probe] synchronize #2 returned", flush=True)

    # A third round: if each synchronize costs a stream, we should see a NEW
    # stream address per launch rather than one stable handle.
    print("[probe] === launch 3 ===", flush=True)
    ctx.enqueue_function[_bump](lt, grid_dim=1, block_dim=1)
    ctx.synchronize()
    print("[probe] synchronize #3 returned", flush=True)

    print("[probe] done. Read the [intercept] lines against these markers:",
          flush=True)
    print("[probe]   destroys present -> MAX's own behaviour, not ours",
          flush=True)
    print("[probe]   no destroys      -> CUDAGraph provokes them; our bug",
          flush=True)
