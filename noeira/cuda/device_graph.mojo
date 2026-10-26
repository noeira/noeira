"""Record-and-replay a device step on MAX's own `DeviceGraph`.

The replacement for `noeira/cuda/graph.mojo`. Same job, none of the shim:
MAX 26.5 ships `DeviceGraph` / `DeviceGraphBuilder` in `max.gpu.host`, so we
no longer discover MAX's internal stream by `dlsym` interposition and call
`cuStreamBeginCapture` on a stream we do not own.

Measured on an RTX 5090 by `benchmarks/bench_device_graph_spike.mojo`: a
`recording_context()` records our LayoutTensor kernels AND MAX library calls
unmodified, `build` executes nothing, and replay reproduces the work exactly.
Read that file's header before changing anything here.

## What this deletes

`graph.mojo`'s own comments are the bill for borrowing a stream:
`cuStreamDestroy` on our handle, driver SIGSEGVs in `BeginCapture` /
`EndCapture` / `GetCaptureInfo`, a load-bearing `_ctx` field, capture that
"succeeds" with ZERO nodes and says nothing, and
[[_the_preload_is_a_property_of_the_process_not_the_binary]] — a binary you
BUILD and run directly silently loses capture entirely, because `LD_PRELOAD`
comes from pixi's nvidia activation and not from the binary. None of that
exists here. There is no interceptor, no preload, and no `.so` to build.

## The one thing callers must change

`STEP` takes a `DeviceContext` **argument** instead of capturing one. That is
not cosmetic and it cannot be avoided: recording happens because operations are
enqueued through the *recording* context, so a step that reaches for its own
`self.ctx` records nothing and runs eagerly instead — silently, since the work
is still correct. Thread the argument all the way down to the enqueue calls.

## A view built into a local and captured by `STEP` is DEAD

Mojo destroys a value at its LAST USE, and **a mention inside a closure that is
only ever passed as a COMPTIME PARAMETER does not count as one.** So this is
wrong, and wrong silently:

    var lt = LayoutTensor[DT, Layout.row_major(1)](buf)   # <- dies HERE
    def _step(gctx: DeviceContext) capturing raises -> None:
        gctx.enqueue_function[_bump](lt, grid_dim=1, block_dim=1)

Every launch then writes through a dead view. Nothing raises, the graph still
records, and the counter comes back 0. The first draft of
`tests/cuda/test_device_graph_minimal.mojo` did exactly this.

⚠ **THE COMPILER TELLS YOU: `assignment to 'lt' was never used`.** In a file
whose very next line hands `lt` to a kernel, that reads like a false positive.
It is not — it is the whole bug.

Build views INSIDE the step, or capture something whose lifetime is anchored
elsewhere. The production shape is the same rule one level up: **one struct owns
the context and the buffers, and `STEP` mentions only that struct and calls a
method on it.** Every crash in this area came from skipping it.

## Fallback is a RUNTIME latch, not a comptime no-op — and that difference bites

`CUDAGraph` was `comptime if default_accelerator().is_nvidia_gpu()`, so off NVIDIA it
compiled away and `maybe_capture_replay` ran `STEP` every call. `DeviceGraph`
does not compile away: it links everywhere and RAISES at runtime.

    "Graph capture is currently implemented for CUDA and HIP devices only.
     Creating a graph on any other device, such as an Apple GPU or a CPU,
     raises."                              — MAX API docs for `device_graph`

(On an M1 that surfaces as `createGraphBuilder() not supported on this device
context`, and a CPU `DeviceContext` raises too — so this is not an Apple-only
concern, CI without a GPU hits it as well.)

So the first `create` is wrapped and its outcome LATCHED: on a raise the slot
goes permanently `disabled` and every later call runs `STEP` directly. A
comptime gate would need to enumerate the supported backends and would be wrong
the moment Modular adds one; attempting once and remembering cannot be.

⚠ **DO NOT RECORD A STEP THAT CONTAINS A MAX-ALLOCATED GEMM WORKSPACE.**
Measured, same spike, same card: `multistage_gemm`'s split-K workspace and
`matmul_vendor`'s 32MB scratch are allocated through
`ctx.enqueue_create_buffer`, which `recording_context` FORWARDS to the backing
context, and freed when the call returns while the recorded node keeps the raw
pointer. The 512KB split-K case replays with the RIGHT ANSWER while silently
writing over whatever the allocator handed the block to next; the 32MB vendor
case is a hard `CUDA_ERROR_ILLEGAL_ADDRESS` on the FIRST replay, and the error
is sticky. Neither raises at record time. Before recording a step, check it
with `-D LOGGING_LEVEL=INFO`:

    | grep -c 'K partitions: [2-9]'          # MAX-dispatched split-K
    | grep -c 'Executing: vendor BLAS'       # 32MB scratch per call

Both must be 0. Our own `nn/core/splitk_gemm.mojo` takes a caller-owned
workspace and is safe by construction; it is MAX's dispatch that is not.

## !! AND THAT RULES OUT MOST RL STEPS TODAY — MEASURED, NOT FEARED

`multi_gemm_cond` (matmul/gpu/__init__.mojo:591) is

    m > 1 and n % 128 == 0 and k % 32 == 0 and k >= 128

so a `Linear` whose INPUT width is under 128, or whose OUTPUT width is not a
multiple of 128, goes to `matmul_vendor` and its 32MB-per-call scratch. **That
is the first and last layer of every actor and critic we have.** SAC on
Pendulum (`OBS=3, ACT=1, H=128`) hits it four ways:

    actor  L1   [256,3] @ [3,128]     k=3    -> vendor
    critic L1   [256,4] @ [4,128]     k=4    -> vendor
    actor  head 128 -> 2              n=2    -> vendor
    critic head 128 -> 1              n=1    -> vendor

Recording that step dies with `CUDA_ERROR_ILLEGAL_ADDRESS` a few hundred
updates in. Under STREAM capture the same allocation RAISES, MAX catches it at
matmul/gpu/__init__.mojo:1373 and drops to `matmul_kernel_naive` — correct,
slower, alive. **So for any step containing a vendor-path GEMM, stream capture
is strictly SAFER than recording, and small obs/act dims make that nearly every
deep-RL step.**

⚠ **THEREFORE `noeira/cuda/graph.mojo` CANNOT BE RETIRED YET.** The mechanism
here works — `benchmarks/bench_device_graph_spike.mojo` arms A/B/C pass and
`tests/cuda/test_device_graph_minimal.mojo` records on a 5090 — but real
training steps hit MAX's allocator. This module stays behind
`NOEIRA_GRAPH_BACKEND=device` as dormant infrastructure until either MAX
allocates through `DeviceGraphBuilder.create_buffer` (the API its own docstring
tells callers to use), or every GEMM in the step is routed off the vendor path.

⚠ **CHECK THE CANDIDATE, DO NOT REASON FROM "its shapes look fine".** SAC was
picked as the first migration precisely because its MLP GEMMs "never hit
split-K or the vendor path". The split-K half was right; the vendor half was
never checked, and the two greps above would have taken one run.
"""

from std.os import getenv

from max.gpu.host import DeviceContext, DeviceGraph, DeviceGraphBuilder


struct GraphSlot(Movable):
    """Caller-owned record/replay state: the graph, or a latched refusal.

    Two fields rather than one `Optional`, because "not recorded yet" and
    "recording is impossible here" need opposite handling and collapsing them
    costs a `create` attempt — plus its printed diagnostic — on EVERY call.
    `CUDAGraph` learned the same thing: it stored the disabled graph so later
    calls stopped retrying construction.
    """

    var graph: Optional[DeviceGraph]
    var disabled: Bool

    def __init__(out self):
        self.graph = None
        self.disabled = False

    def __init__(out self, *, deinit move: Self):
        self.graph = move.graph^
        self.disabled = move.disabled

    def is_recorded(self) -> Bool:
        return Bool(self.graph)

    def is_disabled(self) -> Bool:
        return self.disabled


def maybe_record_replay[
    STEP: def (DeviceContext) capturing raises -> None,
](mut slot: GraphSlot, ctx: DeviceContext) raises:
    """Record `STEP` into `slot` on first call; replay it thereafter.

    `STEP` must enqueue the SAME kernel sequence every call — the recorded
    graph holds raw pointers, so a step whose shapes or buffers vary between
    calls cannot be replayed. Device-side RNG is fine (and intended): each
    replay advances the counter, so a captured sampling step still draws a
    fresh minibatch.

    ⚠ EVERY BUFFER `STEP` TOUCHES MUST OUTLIVE THE LAST REPLAY, not the
    recording. A graph holds raw device pointers; Mojo destroys a value at its
    LAST USE. One struct that owns the context and the buffers, with `STEP`
    mentioning only that struct, is the shape that makes this true by
    construction — and the shape every crash in this area came from skipping.

    ⚠ HOST BOOKKEEPING DOES NOT BELONG IN `STEP`. Step counters and metric
    cadence live in the caller's loop, advanced once per logical update, so
    they stay correct whether the step ran directly or via replay. Host work
    inside `STEP` is recorded ONCE and then never happens again.

    On the first call `STEP` runs once — to settle the stream and let module
    loads, autotuning and any first-call allocation happen OUTSIDE the
    recording — and the graph is then built from a second, non-executing pass.
    So the first call performs exactly ONE logical update, same as
    `maybe_capture_replay` did.

    `NOEIRA_DEVICE_GRAPH=0` disables recording without a rebuild, for
    bisecting a suspected graph problem against a known-good run.
    """
    # ⚠ `flush=True` on every diagnostic in this file. Mojo's stdout is BLOCK
    # buffered, so a step that hangs shows you nothing it printed and the last
    # line you saw is not where it stopped. That cost a debugging round.
    comptime TRACE_VAR = "NOEIRA_DEVICE_GRAPH_TRACE"
    var trace = getenv(TRACE_VAR, "0") == "1"

    if slot.disabled:
        if trace:
            print("[DeviceGraph] latched-disabled -> STEP directly", flush=True)
        STEP(ctx)
        return

    if slot.graph:
        # ⚠ `replay()` SUBMITS; it does not wait. Ordering against work the
        # caller enqueues before or after is the context's, same as any other
        # enqueue — do not add a synchronize here, the caller's loop owns that.
        if trace:
            print("[DeviceGraph] replay ...", flush=True)
        slot.graph.value().replay()
        if trace:
            print("[DeviceGraph] replay submitted", flush=True)
        return

    if getenv("NOEIRA_DEVICE_GRAPH", "1") == "0":
        slot.disabled = True
        print(
            "[DeviceGraph] disabled by NOEIRA_DEVICE_GRAPH=0 — running steps"
            " directly (correct, no replay speedup).",
            flush=True,
        )
        STEP(ctx)
        return

    # Warm-up OUTSIDE the recording: first-call autotuning, module loads and
    # one-off allocations happen here rather than becoming recorded nodes.
    if trace:
        print("[DeviceGraph] warm-up STEP ...", flush=True)
    STEP(ctx)
    ctx.synchronize()
    if trace:
        print("[DeviceGraph] warm-up done; recording ...", flush=True)

    # ⚠ `build` RECORDS, it does not execute. Nothing in here runs until
    # `replay()`, which is why the warm-up above is the first call's update.
    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        with builder.recording_context() as rec:
            STEP(rec)

    try:
        slot.graph = DeviceGraph.create(ctx, build)
        if trace:
            print("[DeviceGraph] recorded", flush=True)
    except e:
        # LATCHED. See the module docstring: DeviceGraph links on every
        # platform and raises on the unsupported ones, so this is the ordinary
        # Apple / CPU path and not an error. `STEP` already ran for this call,
        # so returning here is a completed update, not a skipped one.
        slot.disabled = True
        print(
            "[DeviceGraph] recording unavailable here — running steps directly"
            " (correct results, no replay speedup). Reason: " + String(e),
            flush=True,
        )
