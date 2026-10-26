"""Where does a `DeviceGraph` step actually hang? A numbered, flushed trace.

    pixi run -e nvidia mojo build -I . -o /tmp/probe_dg \\
        tests/cuda/probe_device_graph_steps.mojo
    /tmp/probe_dg                    # bare: no pixi, so no LD_PRELOAD

⚠ **`flush=True` ON EVERY PRINT IS LOAD-BEARING** — the same note
`probe_max_stream_lifetime.mojo:33` carries. Mojo's stdout is block-buffered,
so a hung process shows you NOTHING it printed, and the last line you see is
not where it stopped. `test_device_graph_minimal.mojo` hung on a 5090 printing
only the interceptor's own (C, line-buffered) banner, which made it look like
the hang was before any Mojo code ran. It was not; the output was simply still
in the buffer. `stdbuf -oL` does not help either — that sets the C runtime's
mode, and Mojo does its own buffering above it.

⚠ **NO `TestSuite` HERE, ON PURPOSE.** A hang inside a test harness cannot be
told apart from a hang in the harness itself, and `discover_tests` instantiates
every top-level def for the host before running anything. A plain `main` with
numbered stages removes both variables.

## Reading it

The trace bisects the whole path, and where it stops names the culprit:

    stops before [1]   not this file — check for a stale process holding the
                       GPU (`nvidia-smi`), which is easy to have after a
                       Ctrl-C'd profile run. An absent shell prompt is not a
                       dead run.
    stops at [1]       DeviceContext creation. Environment, not graphs.
    stops at [3]       a BARE kernel launch + synchronize, no graph anywhere.
                       Also not graphs — JIT/ptxas or a wedged device.
    stops at [6]       `DeviceGraph.create` — recording itself.
    stops at [9]       `replay()`.
    stops at [11]      teardown: releasing the graph, or freeing a buffer the
                       graph still points at.

Stage [3] is the important one. It uses no graph API at all, so if the trace
stops there the problem was never in `device_graph.mojo` and every conclusion
drawn from "the graph test hangs" is about the machine instead.
"""

from max.gpu import thread_idx
from std.os import getenv
from noeira.core.tiles import Tile, tile
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.cuda import GraphSlot, maybe_record_replay


def main() raises:
    print("[probe] === start ===", flush=True)

    print("[probe] [1] DeviceContext() ...", flush=True)
    var ctx = DeviceContext()
    print("[probe] [1] ok — device:", ctx.name(), flush=True)

    print("[probe] [2] create + fill buffer ...", flush=True)
    var buf = ctx.enqueue_create_buffer[DT](1)
    buf.enqueue_fill(Scalar[DT](0))
    ctx.synchronize()
    print("[probe] [2] ok", flush=True)

    @always_inline
    def _bump(b: Tile[DT, 1]):
        if Int(thread_idx.x) != 0:
            return
        b[0] = b[0] + Scalar[DT](1.0)

    # ⚠ STAGE 3 TOUCHES NO GRAPH API. If the trace dies here the machine or the
    # toolchain is the problem and `device_graph.mojo` is exonerated. This is
    # also the first JIT of `_bump`, so on NVIDIA it pays ptxas.
    print("[probe] [3] bare enqueue_function + synchronize (NO graph) ...", flush=True)
    var v0 = tile[1](buf)
    ctx.enqueue_function[_bump](v0, grid_dim=1, block_dim=1)
    ctx.synchronize()
    print("[probe] [3] ok", flush=True)

    print("[probe] [4] read back ...", flush=True)
    var h = ctx.enqueue_create_host_buffer[DT](1)
    ctx.enqueue_copy(h, buf)
    ctx.synchronize()
    print("[probe] [4] ok — counter =", Float64(h[0]), "(want 1.0)", flush=True)

    print("[probe] [5] GraphSlot() ...", flush=True)
    var slot = GraphSlot()
    print(
        "[probe] [5] ok — NOEIRA_DEVICE_GRAPH =",
        getenv("NOEIRA_DEVICE_GRAPH", "1"),
        flush=True,
    )

    # The view is built INSIDE the step: a value whose only mention is inside a
    # closure passed as a comptime parameter is destroyed at its construction
    # line. See `device_graph.mojo`'s header.
    def _step(gctx: DeviceContext) capturing raises -> None:
        var v = tile[1](buf)
        gctx.enqueue_function[_bump](v, grid_dim=1, block_dim=1)

    print("[probe] [6] maybe_record_replay #1 (warm-up + record) ...", flush=True)
    maybe_record_replay[_step](slot, ctx)
    print(
        "[probe] [6] ok — recorded =",
        slot.is_recorded(),
        " disabled =",
        slot.is_disabled(),
        flush=True,
    )

    print("[probe] [7] synchronize ...", flush=True)
    ctx.synchronize()
    print("[probe] [7] ok", flush=True)

    print("[probe] [8] read back ...", flush=True)
    ctx.enqueue_copy(h, buf)
    ctx.synchronize()
    print("[probe] [8] ok — counter =", Float64(h[0]), "(want 2.0)", flush=True)

    print("[probe] [9] maybe_record_replay #2 (replay) ...", flush=True)
    maybe_record_replay[_step](slot, ctx)
    print("[probe] [9] ok", flush=True)

    print("[probe] [10] synchronize + read ...", flush=True)
    ctx.synchronize()
    ctx.enqueue_copy(h, buf)
    ctx.synchronize()
    print("[probe] [10] ok — counter =", Float64(h[0]), "(want 3.0)", flush=True)

    # ⚠ ORDER MATTERS. The graph holds RAW POINTERS into `buf`, so the slot is
    # released FIRST and the buffer only after. The reverse order frees memory
    # a live graph still references.
    print("[probe] [11] teardown: release slot, then buffer ...", flush=True)
    _ = slot^
    print("[probe] [11a] slot released", flush=True)
    _ = h^
    _ = buf^
    print("[probe] [11] ok", flush=True)

    print("[probe] === done ===", flush=True)
