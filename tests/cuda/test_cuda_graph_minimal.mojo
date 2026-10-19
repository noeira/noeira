"""Minimal CUDA-graph gate: does capture/replay still work AT ALL?

⚠⚠ THIS EXISTS BECAUSE CAPTURE CAN FAIL SILENTLY AND NOTHING WOULD SAY SO.
`CUDAGraph` does not talk to the CUDA driver directly — every call goes
through `noeira/cuda/libcuda_intercept.so`, an LD_PRELOAD shim that hooks
CUDA via `dlsym` and DISCOVERS MOJO'S INTERNAL STREAM by watching which
stream Mojo enqueues on (hence "requires at least one prior kernel launch
for stream discovery" in `CUDAGraph.__init__`).

That contract is with Mojo's INTERNALS, not a public API, and nothing about
it is compile-checked. It breaks if a Mojo upgrade changes stream handling,
or if the runtime moves its driver calls behind `cuGetProcAddress` (which
modern CUDA prefers) where a `dlsym` hook cannot see them. The failure is not
an error: `cuStreamBeginCapture` succeeds on a stream nothing is enqueued to,
`cuStreamEndCapture` returns an EMPTY graph, and replaying it does nothing —
or faults. `maybe_capture_replay`'s own comment predicts exactly this ("a
tiny/zero node count means capture silently failed").

The two existing capture tests (`test_dqn_gpu_batched_capture_smoke`,
`test_dreamerv3_capture_parity`) are agent-level smokes: they take minutes,
they drag in networks and replay buffers, and if they fail you cannot tell
capture from the agent. This file is the primitive underneath them — one
1-element buffer and one increment kernel, seconds to run — so that "is the
graph harness alive?" is answerable on its own.

⚠ ASSERTING `num_nodes() > 0` IS THE POINT. A test that only checks the
replayed VALUE can pass while capturing nothing, because the settle run in
`maybe_capture_replay` executes the work for real. Node count is the only
direct evidence that the graph contains anything.

Platform contract, both halves gated here:
  NVIDIA      real capture; the graph holds nodes; replay executes it.
  non-NVIDIA  every method is a comptime no-op and `maybe_capture_replay`
              runs STEP on EVERY call — bit-identical to not using graphs.
              That is what makes Apple a valid correctness (not performance)
              check, so it is worth pinning rather than assuming.

⚠⚠ IF THIS CRASHES OR ABORTS, SUSPECT A DEAD DeviceContext FIRST.
Mojo destroys a value at its LAST USE, not at end of scope. If a caller
writes `var g = CUDAGraph(ctx)` and never mentions `ctx` again, the context
is destroyed the moment `__init__` returns — and MAX's destructor
SYNCHRONIZES AND DESTROYS the stream being captured. `CUDAGraph` now stores
the context to make that impossible, but the same trap applies to anything
holding a borrowed FFI handle.

That single mechanism produced every symptom of the 2026-08-09 arc, and each
one invited a different wrong theory:

    cuStreamDestroy on our handle           the destructor, not stream pooling
    cuStreamSynchronize "during capture"    the destructor, not MAX bookkeeping
    SIGSEGV in BeginCapture / EndCapture /  use-after-free on a freed stream —
      GetCaptureInfo_v2                     undefined, hence INCONSISTENT
                                            (rc=0 here, a fault there)

⚠ INCONSISTENCY ACROSS ENTRY POINTS IS THE TELL. Several APIs failing several
different ways is far more likely to be ONE dangling handle than several
bugs. Chasing them separately cost eight rounds here.

⚠ `MODULAR_DEBUG=device-sync-mode` CANNOT DEBUG CAPTURE. It synchronizes after
every device op, and a synchronize during capture is illegal — so it
manufactures a CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED of its own.

⚠ `"CUDA call failed: ..."` is MAX's error formatter; ours are tagged
`[CUDAGraph]`. Read the prefix before assuming the fault is in this file.

Run with:
    pixi run -e nvidia mojo run -I . tests/cuda/test_cuda_graph_minimal.mojo
    pixi run -e apple  mojo run -I . tests/cuda/test_cuda_graph_minimal.mojo

    # bisect a suspected capture problem against a no-graph run:
    NOEIRA_CUDA_GRAPH=0 pixi run -e nvidia mojo run -I . \
        tests/cuda/test_cuda_graph_minimal.mojo

    # full driver tracing (launches, stream lifetime, capture rc):
    NOEIRA_INTERCEPT_LOG=1 pixi run -e nvidia mojo run -I . \
        tests/cuda/test_cuda_graph_minimal.mojo
"""

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext, DeviceBuffer
from std.sys import default_accelerator
from std.testing import assert_true, TestSuite
from noeira.core.tiles import Tile, tile

from noeira.nn.constants import DT
from noeira.cuda import CUDAGraph, maybe_capture_replay

comptime REPLAYS = 5


def _read(ctx: DeviceContext, buf: DeviceBuffer[DT]) raises -> Float64:
    var h = ctx.enqueue_create_host_buffer[DT](1)
    ctx.enqueue_copy(h, buf)
    ctx.synchronize()
    return Float64(h[0])


def test_capture_records_nodes() raises:
    """`end_capture` produced a graph with at least one node.

    This is the assertion that catches a broken interceptor. If stream
    discovery picks the wrong stream, begin/end capture BOTH return success
    and the node count is 0 — no error anywhere.
    """
    # ⚠ NESTED, not module-level. `TestSuite.discover_tests[
    # __functions_in_module()]` instantiates every top-level def for the HOST,
    # and a GPU kernel fails there with "target does not support operation:
    # _get_intrinsic_name". Nesting keeps it out of discovery — the same shape
    # `Phyics3dBatchedEnv` uses for its kernels.
    @always_inline
    def _bump_kernel(
        buf: Tile[DT, 1],
    ):
        """`buf[0] += 1`. One thread — the smallest thing a graph can hold."""
        if Int(thread_idx.x) != 0:
            return
        buf[0] = buf[0] + Scalar[DT](1.0)

    var ctx = DeviceContext()
    var buf = ctx.enqueue_create_buffer[DT](1)
    var lt = tile[1](buf)

    # ⚠ Stream discovery needs a prior launch — see `CUDAGraph.__init__`.
    # Constructing the graph before any kernel has run is itself a way to get
    # a silent zero-node capture.
    ctx.enqueue_function[_bump_kernel](lt, grid_dim=1, block_dim=1)
    ctx.synchronize()

    var g = CUDAGraph(ctx)

    # Only reachable via NOEIRA_CUDA_GRAPH=0, which exists so a suspected
    # capture problem can be bisected against a known-good run without a
    # rebuild. Capture is ON by default and the assertion below is the real
    # gate. (This branch briefly WAS the default, while we wrongly believed
    # MAX destroyed its stream on every synchronize.)
    if g.is_disabled():
        print("  CUDAGraph DISABLED (no live Mojo stream) — capture skipped")
        print("  correctness is gated by test_maybe_capture_replay_lifecycle")
        return

    g.begin_capture()
    ctx.enqueue_function[_bump_kernel](lt, grid_dim=1, block_dim=1)
    g.end_capture()

    print("  captured nodes =", g.num_nodes())

    comptime if default_accelerator().is_nvidia_gpu():
        assert_true(
            g.num_nodes() > 0,
            "CUDA graph captured ZERO nodes. begin/end capture both"
            + " succeeded, so this is not a CUDA error — it means the"
            + " interceptor recorded no launches on the captured stream."
            + " Check (a) that the DeviceContext is still alive — Mojo"
            + " destroys it at its LAST USE and MAX's destructor takes the"
            + " stream with it, which is what broke this in 2026-08; and"
            + " (b) that libcuda_intercept.so still sees the launches, i.e."
            + " that MAX has not moved them past both the dlsym and"
            + " cuGetProcAddress hooks.",
        )
    else:
        assert_true(
            g.num_nodes() == 0,
            "non-NVIDIA CUDAGraph should be a comptime no-op with no nodes.",
        )


def test_replay_executes_the_captured_work() raises:
    """Each replay runs the captured kernel exactly once.

    ⚠ THE ARITHMETIC IS THE ASSERTION. Capture ENQUEUES BUT DOES NOT EXECUTE,
    so after `begin/end capture` the counter must be unchanged, and each
    replay must add exactly 1. A harness that silently executed during
    capture, or replayed nothing, lands on a different number.
    """
    # ⚠ NESTED, not module-level. `TestSuite.discover_tests[
    # __functions_in_module()]` instantiates every top-level def for the HOST,
    # and a GPU kernel fails there with "target does not support operation:
    # _get_intrinsic_name". Nesting keeps it out of discovery — the same shape
    # `Phyics3dBatchedEnv` uses for its kernels.
    @always_inline
    def _bump_kernel(
        buf: Tile[DT, 1],
    ):
        """`buf[0] += 1`. One thread — the smallest thing a graph can hold."""
        if Int(thread_idx.x) != 0:
            return
        buf[0] = buf[0] + Scalar[DT](1.0)

    var ctx = DeviceContext()
    var buf = ctx.enqueue_create_buffer[DT](1)
    buf.enqueue_fill(Scalar[DT](0))
    var lt = tile[1](buf)

    ctx.enqueue_function[_bump_kernel](lt, grid_dim=1, block_dim=1)
    ctx.synchronize()
    var after_warm = _read(ctx, buf)

    var g = CUDAGraph(ctx)
    if g.is_disabled():
        print("  CUDAGraph DISABLED (no live Mojo stream) — replay skipped")
        return

    g.begin_capture()
    ctx.enqueue_function[_bump_kernel](lt, grid_dim=1, block_dim=1)
    g.end_capture()
    ctx.synchronize()
    var after_capture = _read(ctx, buf)

    comptime if default_accelerator().is_nvidia_gpu():
        assert_true(
            after_capture == after_warm,
            "the counter moved during CAPTURE (" + String(after_warm)
            + " -> " + String(after_capture)
            + "). Capture must enqueue without executing; if it executed,"
            + " the stream was never in capture mode.",
        )
        for _ in range(REPLAYS):
            g.replay_on_mojo_stream()
        ctx.synchronize()
        var after_replay = _read(ctx, buf)
        print(
            "  warm =", after_warm, " after capture =", after_capture,
            " after", REPLAYS, "replays =", after_replay,
        )
        assert_true(
            after_replay == after_warm + Float64(REPLAYS),
            "expected " + String(after_warm + Float64(REPLAYS))
            + " after " + String(REPLAYS) + " replays, got "
            + String(after_replay)
            + ". A short count means the graph is empty or partial; an over"
            + "-count means work executed during capture as well.",
        )
    else:
        # ⚠ NON-NVIDIA IS NOT "NOTHING HAPPENS". `begin_capture` /
        # `end_capture` are comptime no-ops, but the kernel BETWEEN them is an
        # ordinary launch and RUNS IMMEDIATELY — so the counter advances by one
        # at "capture" time and never again on replay. (Measured: asserting
        # the counter was unchanged here failed on Apple.)
        #
        # The consequence is worth stating: on non-NVIDIA, a CUDAGraph driven
        # DIRECTLY does its work once, at capture, and every `replay()` is a
        # silent no-op. Only `maybe_capture_replay` is safe on both platforms,
        # because it runs the closure on every call. Anything reaching for
        # `CUDAGraph` by hand has to know this.
        print("  non-NVIDIA: capture is a no-op; the kernel ran inline")
        assert_true(
            after_capture == after_warm + 1.0,
            "non-NVIDIA: the kernel between begin/end capture should have run"
            + " inline (expected " + String(after_warm + 1.0) + ", got "
            + String(after_capture) + ").",
        )
        for _ in range(REPLAYS):
            g.replay_on_mojo_stream()
        ctx.synchronize()
        assert_true(
            _read(ctx, buf) == after_capture,
            "non-NVIDIA replay should add nothing.",
        )


def test_maybe_capture_replay_lifecycle() raises:
    """The harness the trainers actually use, on a counter.

    `maybe_capture_replay` settles (RUNS the closure), then captures a second
    run (which does NOT run), then replays on every later call. So after
    `1 + R` calls the counter is `1 + R` on NVIDIA — and `1 + R` on Apple too,
    by a different route (the closure simply runs every call). The two
    platforms agreeing on the VALUE is the property that makes an Apple run a
    meaningful correctness check of a graph-enabled trainer.
    """
    # ⚠ NESTED, not module-level. `TestSuite.discover_tests[
    # __functions_in_module()]` instantiates every top-level def for the HOST,
    # and a GPU kernel fails there with "target does not support operation:
    # _get_intrinsic_name". Nesting keeps it out of discovery — the same shape
    # `Phyics3dBatchedEnv` uses for its kernels.
    @always_inline
    def _bump_kernel(
        buf: Tile[DT, 1],
    ):
        """`buf[0] += 1`. One thread — the smallest thing a graph can hold."""
        if Int(thread_idx.x) != 0:
            return
        buf[0] = buf[0] + Scalar[DT](1.0)

    var ctx = DeviceContext()
    var buf = ctx.enqueue_create_buffer[DT](1)
    buf.enqueue_fill(Scalar[DT](0))
    var lt = tile[1](buf)

    ctx.enqueue_function[_bump_kernel](lt, grid_dim=1, block_dim=1)
    ctx.synchronize()

    var graph = Optional[CUDAGraph](None)

    def _step() capturing raises -> None:
        ctx.enqueue_function[_bump_kernel](lt, grid_dim=1, block_dim=1)

    for _ in range(1 + REPLAYS):
        maybe_capture_replay[_step](graph, ctx)
    ctx.synchronize()

    var got = _read(ctx, buf)
    var want = 1.0 + 1.0 + Float64(REPLAYS)  # warm + settle + R replays
    print("  maybe_capture_replay: counter =", got, " expected =", want)
    assert_true(
        got == want,
        "maybe_capture_replay ran the work " + String(got)
        + " times, expected " + String(want)
        + ". On NVIDIA a shortfall of exactly REPLAYS means the captured"
        + " graph is empty (every replay a no-op) — the silent-capture"
        + " failure this file exists to catch.",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
