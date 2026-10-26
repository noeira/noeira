"""Minimal gate for `noeira/cuda/device_graph.mojo`: record, replay, latch.

The successor to `test_cuda_graph_minimal.mojo`, and much smaller, because
most of what that file gates does not exist here: there is no interceptor to
be unhooked, no `LD_PRELOAD` to be missing, no borrowed stream to be destroyed
under us, and no silent zero-node capture. `DeviceGraph` is a supported API
that either records or raises.

⚠ **ONE TEST ON ONE `DeviceContext`, DELIBERATELY.** The two-test version of
this file HUNG on a 5090 while `probe_device_graph_steps.mojo` — the same
calls, a plain `main`, no harness — ran clean end to end (`recorded = True`,
counter 1 -> 2 -> 3). The only differences were `TestSuite` and a SECOND
`DeviceContext` constructed after the first had been destroyed. If this file
ever hangs again, run that probe FIRST: it separates "the graph API is broken"
from "the harness is", and last time the answer was the harness.

## The arithmetic is the assertion, and it holds on BOTH platforms

N calls to `maybe_record_replay` must produce EXACTLY N increments, whichever
path ran:

    NVIDIA   call 1 runs STEP once (the warm-up) and RECORDS a second,
             non-executing pass; calls 2..N replay.        1 + (N-1) = N
    Apple    `DeviceGraph.create` raises, the slot latches disabled, and
             STEP runs directly every call.                    N

That equality is the whole contract — the caller's loop must not care which
happened — and it makes a Mac a real correctness check rather than a build
check. It also catches the two failures that would otherwise be silent: a
`build` that EXECUTES lands on N+1, and a replay that runs nothing lands on 1.

⚠ `test_platform_contract` is what stops the arithmetic test passing
vacuously. On Apple every arm of this file can pass with nothing recorded at
all, so the platform contract is pinned separately: NVIDIA must end up
RECORDED, everything else DISABLED. Without it, "N increments" only proves
`STEP` was called N times, which the disabled path does by definition.

Run with:
    pixi run -e nvidia mojo run -I . tests/cuda/test_device_graph_minimal.mojo
    pixi run -e apple  mojo run -I . tests/cuda/test_device_graph_minimal.mojo

    # if it hangs, this is the bisector — 11 numbered, FLUSHED stages, no
    # TestSuite. Mojo's stdout is BLOCK buffered, so a hung run shows you
    # nothing it printed and the last line you saw is not where it stopped.
    pixi run -e nvidia mojo build -I . -o /tmp/probe_dg \
        tests/cuda/probe_device_graph_steps.mojo && /tmp/probe_dg

    # bisect a suspected graph problem against a known-good run:
    NOEIRA_DEVICE_GRAPH=0 pixi run -e nvidia mojo run -I . \
        tests/cuda/test_device_graph_minimal.mojo
"""

from max.gpu import thread_idx
from std.sys import default_accelerator
from std.testing import assert_true, TestSuite
from noeira.core.tiles import Tile, tile
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.cuda import GraphSlot, maybe_record_replay

comptime CALLS = 5


def _read(ctx: DeviceContext, buf: DeviceBuffer[DT]) raises -> Float64:
    var h = ctx.enqueue_create_host_buffer[DT](1)
    ctx.enqueue_copy(h, buf)
    ctx.synchronize()
    return Float64(h[0])


def test_record_replay_contract() raises:
    """Both halves of the contract, on ONE `DeviceContext`.

    ⚠ DELIBERATELY ONE TEST, NOT TWO. The two-test version hung on a 5090
    while `probe_device_graph_steps.mojo` — the same calls, a plain `main`,
    no harness — ran clean end to end. The only differences were `TestSuite`
    and a SECOND `DeviceContext` built after the first had been destroyed.
    Merging removes both, and costs nothing: the two assertions were always
    about the same run.

    ⚠ AND THEY BELONG TOGETHER ANYWAY. The arithmetic alone is satisfied
    trivially by the disabled path — `STEP` called N times increments N times
    whether or not anything was ever recorded — so a NVIDIA regression that
    silently stopped recording would still pass it. The platform assertion is
    what makes the count mean something, so asserting them on the same slot
    is stronger than asserting them on two.
    """
    # ⚠ NESTED, not module-level. `TestSuite.discover_tests` instantiates every
    # top-level def for the HOST, and a GPU kernel fails there with "target
    # does not support operation: _get_intrinsic_name".
    @always_inline
    def _bump(buf: Tile[DT, 1]):
        if Int(thread_idx.x) != 0:
            return
        buf[0] = buf[0] + Scalar[DT](1.0)

    var ctx = DeviceContext()
    var buf = ctx.enqueue_create_buffer[DT](1)
    buf.enqueue_fill(Scalar[DT](0))
    ctx.synchronize()

    # ⚠⚠ THE VIEW IS BUILT INSIDE THE STEP, NOT CAPTURED FROM A LOCAL, AND THE
    # FIRST DRAFT OF THIS TEST GOT IT WRONG.
    #
    #     var lt = LayoutTensor[DT, Layout.row_major(1)](buf)   # <- WRONG
    #     def _step(...): gctx.enqueue_function[_bump](lt, ...)
    #
    # Mojo destroys a value at its LAST USE, and a mention inside a closure
    # that is only ever passed as a COMPTIME PARAMETER does not count as one.
    # So `lt` died on its own construction line, every launch wrote through a
    # dead view, and the counter came back 0.0 with nothing raised anywhere.
    # `buf` is safe to capture precisely because `_read` mentions it below.
    #
    # ⚠ THE COMPILER TELLS YOU: "assignment to 'lt' was never used". In a file
    # whose next line hands `lt` to a kernel that warning reads like a false
    # positive. It is not — it is the whole bug.
    #
    # ⚠ AND THE ARGUMENT IS THE POINT. A step that reaches for an outer `ctx`
    # instead of using `gctx` enqueues EAGERLY during the recording pass: the
    # work is still correct, the graph is still built, and it records NOTHING.
    # Nothing raises. The counter is what tells you — it lands on CALLS+1.
    def _step(gctx: DeviceContext) capturing raises -> None:
        var v = tile[1](buf)
        gctx.enqueue_function[_bump](v, grid_dim=1, block_dim=1)

    var slot = GraphSlot()
    for _ in range(CALLS):
        maybe_record_replay[_step](slot, ctx)
    ctx.synchronize()

    # N calls, N increments — whichever path ran:
    #   NVIDIA  1 warm-up + (N-1) replays
    #   Apple   N direct runs on a latched-disabled slot
    var got = _read(ctx, buf)
    assert_true(
        got == Float64(CALLS),
        "expected exactly "
        + String(CALLS)
        + " increments, got "
        + String(got)
        + ". "
        + String(CALLS + 1)
        + " means the recording pass EXECUTED (the step enqueued on the outer"
        + " context, not the recording one). 1 means replay ran nothing.",
    )

    # Which path ran. Without this the count above is vacuous.
    comptime if default_accelerator().is_nvidia_gpu():
        assert_true(
            slot.is_recorded() and not slot.is_disabled(),
            "NVIDIA must RECORD. A latched-disabled slot here means"
            " DeviceGraph.create raised — read the printed reason. It is not"
            " an LD_PRELOAD problem: this path has no interceptor.",
        )
    else:
        assert_true(
            slot.is_disabled() and not slot.is_recorded(),
            "off CUDA/HIP, DeviceGraph.create must raise and the slot must"
            " latch disabled. Recording here would mean MAX grew a backend"
            " and this contract needs updating.",
        )

    # ⚠ ORDER MATTERS: SLOT FIRST, BUFFER SECOND. The graph holds RAW POINTERS
    # into `buf`, so releasing the buffer while a live graph still references
    # it is a use-after-free waiting for the next replay. Mojo destroys a value
    # at its LAST USE, so these two lines ARE the ordering.
    _ = slot^
    _ = buf^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
