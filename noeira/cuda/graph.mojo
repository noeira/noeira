"""CUDA Graph capture and replay for Mojo GPU kernels.

On Apple Silicon / non-NVIDIA platforms, all methods are compile-time
no-ops with zero overhead.

Usage:
    from noeira.cuda import CUDAGraph

    var graph = CUDAGraph(ctx)
    graph.begin_capture()
    my_gpu_kernels(ctx, ...)
    graph.end_capture()

    for i in range(1000):
        graph.replay()

Requires: LD_PRELOAD with libcuda_intercept.so.

⚠ THE PRELOAD IS A PROPERTY OF THE PROCESS, NOT THE BINARY. It is set by
pixi's nvidia-environment activation (`pixi.toml [feature.nvidia.activation]`),
so `pixi run -e nvidia mojo run ...` has it and a binary you built and then
invoke DIRECTLY — `./act_profile` — does not. Without it capture disables
itself, because the interceptor hooks by exporting its own `dlsym` (symbol
interposition) and that only works if the library is loaded BEFORE MAX
resolves its CUDA entry points. The `dlopen` this file performs is far too
late to hook anything; it only reaches the entry points.

    LD_PRELOAD=$PWD/noeira/cuda/libcuda_intercept.so ./act_profile

The banner tells you which you have. Preloaded, "[intercept] CUDA interceptor
loaded" is the FIRST line of the run. If it appears in the middle — at the
moment `CUDAGraph` is constructed — the library was only dlopened, its
`g_mojo_stream` is NULL because it never saw a launch, and capture disables.
"""

from std.sys import has_nvidia_gpu_accelerator
from std.ffi import OwnedDLHandle, c_int
from std.memory import alloc, dealloc
from std.os import getenv
from max.gpu.host import DeviceContext
from max.gpu.host._nvidia_cuda import CUDA


comptime _CUptr = Pointer[NoneType, MutUntrackedOrigin]


@always_inline
def _uninit[T: AnyType](out value: T):
    """Returns uninitialized data."""
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(value))



def _print_capture_disabled(launches: Int):
    """Why capture is off, and which fix applies.

    A NULL Mojo stream has TWO causes needing opposite fixes, and the launch
    count separates them: the interceptor records the stream on EVERY
    `cuLaunchKernelEx`, so having seen none means it is not hooked into this
    process at all and the NULL says nothing about MAX's streams.

    ⚠ Module level on purpose, not inlined into the NVIDIA-only `comptime if`
    that calls it. `comptime if` PRUNES, so a message written inside that
    branch is never compiled on a non-NVIDIA machine — which is where this
    file is usually edited. A diagnostic that only typechecks on the hardware
    it diagnoses is a diagnostic you find out is broken at the worst moment.

    ⚠ The message this replaced asserted a single cause — "MAX destroys its
    stream after a synchronize" — which `tests/cuda/probe_max_stream_lifetime`
    had already REFUTED (MAX keeps one stream and tears it down when the
    DeviceContext dies; see `CUDAGraph._ctx`). Stating a stale conclusion as
    fact cost a real debugging session. Say what is observed, name both
    branches, and point at the probe.
    """
    if launches == 0:
        # NOT HOOKED. The interceptor works by exporting its own `dlsym`
        # (symbol interposition), which only takes effect via LD_PRELOAD — a
        # `dlopen` from `CUDAGraph.__init__` is far too late, MAX resolved its
        # CUDA entry points at startup. So this branch means the process
        # started without the preload, which is exactly what running a built
        # binary DIRECTLY does: LD_PRELOAD comes from pixi's nvidia-environment
        # activation, not from the binary.
        print(
            "[CUDAGraph] DISABLED: the CUDA interceptor is not hooked into"
            " this process (0 kernel launches seen)."
        )
        print(
            "    It hooks via LD_PRELOAD, which comes from pixi's nvidia"
            " environment — a binary run directly does not get it. Either run"
            " under `pixi run -e nvidia`, or export it yourself:"
        )
        print(
            "      LD_PRELOAD=$PWD/noeira/cuda/libcuda_intercept.so"
            " ./your_binary"
        )
        print(
            "    Confirm with the banner: preloaded, '[intercept] CUDA"
            " interceptor loaded' is the FIRST line of the run. If it appears"
            " late — at this point — the library was only dlopened and hooks"
            " nothing."
        )
    else:
        # HOOKED, and the stream it last saw is gone: the borrowed-stream
        # problem proper.
        print(
            "[CUDAGraph] DISABLED: the interceptor is hooked ("
            + String(launches) + " launches seen) but the stream it last"
            " recorded has been destroyed, so there is nothing live to"
            " capture."
        )
        print(
            "    Next step is whose destroy that is: NOEIRA_INTERCEPT_LOG=1"
            " pixi run -e nvidia mojo run -I ."
            " tests/cuda/probe_max_stream_lifetime.mojo"
        )
        print(
            "    That probe constructs no CUDAGraph at all, so a destroy"
            " there is MAX's own and a destroy only here is ours to fix."
        )
    print(
        "    Either way the step runs DIRECTLY — correct results, just no"
        " graph-replay speedup."
    )


struct CUDAGraph(Movable):
    """CUDA Graph capture and replay.

    Compile-time gated: all methods are no-ops on non-NVIDIA platforms.
    All CUDA driver calls go through the interceptor library wrappers.
    """

    # 0=init, 1=capturing, 2=captured, 3=DISABLED (no usable Mojo stream)
    var _state: Int
    var _num_nodes: Int
    var _graph: _CUptr
    var _exec: _CUptr
    var _mojo_stream: _CUptr
    var _replay_stream: _CUptr
    var _lib: OwnedDLHandle
    # ⚠⚠ HOLDING THE CONTEXT IS LOAD-BEARING, NOT TIDINESS.
    #
    # Mojo destroys a value at its LAST USE, not at end of scope. A caller
    # that writes
    #
    #     var g = CUDAGraph(ctx)     # <- last mention of ctx
    #     g.begin_capture()
    #
    # has `ctx` destroyed the instant `__init__` returns, and MAX's
    # DeviceContext destructor SYNCHRONIZES AND DESTROYS the stream we are
    # about to capture. Every symptom in the 2026-08-09 arc came from that:
    # `cuStreamDestroy` on our handle, `cuStreamSynchronize` arriving "during
    # capture" (it is the destructor's), and driver SIGSEGVs in
    # cuStreamBeginCapture / EndCapture / GetCaptureInfo (use-after-free on a
    # freed stream, which is undefined and therefore inconsistent — rc=0 from
    # one entry point, a fault from the next).
    #
    # `AsyncRT_DeviceContext_release` was frame #12 of the very first stack
    # trace we looked at. Storing the context makes the graph's lifetime an
    # upper bound on the context's, so no caller can arrange this by accident.
    var _ctx: DeviceContext
    # `intercept_graph_launch` used to be cached here as a raw function
    # pointer. Mojo 1.0's `get_function` returns a `_DLCallable` whose origin
    # is `origin_of(self._lib)` — it deliberately borrows the handle so the
    # library cannot be unloaded mid-call — and that origin cannot be named in
    # a field type, so the callable is resolved per replay instead. `dlsym` on
    # an already-open handle is a hash lookup; against a graph launch plus a
    # stream synchronize it is not measurable.

    def __init__(
        out self, ctx: DeviceContext, *, stream_of_context: Bool = False
    ) raises:
        """Initialize CUDA graph capture.

        On NVIDIA: loads interceptor, discovers Mojo's internal stream.
        Requires at least one prior kernel launch for stream discovery.
        On non-NVIDIA: disabled state, all methods are no-ops.

        `stream_of_context=True` captures `ctx`'s OWN stream, asked of MAX
        (`CUDA(ctx.stream())`, the handle noeira's cuBLASLt path already
        launches on), instead of the interceptor's `g_mojo_stream` — which is
        ONE process-wide value, the stream of whichever kernel launched last.
        With several contexts that is the wrong stream for all but one of them
        (`cuStreamBeginCapture` error 400 in the multi-GPU spike). The caller
        must make `ctx` current (`with ctx.push_context():`) around this
        constructor and every capture/replay call, because the replay stream
        and the instantiated graph belong to the current context. Default off:
        single-context callers keep the discovered stream.
        """
        self._ctx = ctx  # keep the context alive — see the field comment
        self._state = 0
        self._num_nodes = 0
        # Raw _CUptr fields: NVIDIA path overwrites via interceptor calls
        # below; non-NVIDIA paths never read these (compile-time guarded).
        self._graph = _uninit[_CUptr]()
        self._exec = _uninit[_CUptr]()
        self._mojo_stream = _uninit[_CUptr]()
        self._replay_stream = _uninit[_CUptr]()

        comptime if has_nvidia_gpu_accelerator():
            ctx.synchronize()
            self._lib = OwnedDLHandle("./noeira/cuda/libcuda_intercept.so")

            # ⚠ ENABLED AGAIN AS OF THE LIFETIME FIX. This block briefly
            # defaulted OFF, on the conclusion that MAX destroyed and
            # synchronized the stream on its own and the borrowed-stream
            # design was unsalvageable. That conclusion was WRONG. A probe
            # with no CUDAGraph in the process (`probe_max_stream_lifetime`)
            # showed MAX keeps ONE stream across repeated synchronizes and
            # only tears it down when the DeviceContext dies — and the
            # context was dying early because Mojo destroys a value at its
            # LAST USE and `CUDAGraph` did not hold one. See the `_ctx` field.
            #
            # NOEIRA_CUDA_GRAPH=0 disables capture without a rebuild, for
            # bisecting a suspected capture problem against a known-good run.
            if getenv("NOEIRA_CUDA_GRAPH", "1") == "0":
                self._state = 3  # DISABLED by request
                print(
                    "[CUDAGraph] disabled by NOEIRA_CUDA_GRAPH=0 — running"
                    " steps directly (correct, no graph-replay speedup)."
                )
                return

            # ⚠ `get_function`'s parameter is the symbol's RETURN type, and
            # the returned `_DLCallable` IS the callable — there is no unwrap
            # step. This file previously used the pre-1.0 spelling
            # `get_function[def(A, B) thin -> R](name)()`, where the parameter
            # was the whole function type and the trailing `()` unwrapped it.
            # Under 1.0 that spelling still COMPILES — a function type is a
            # valid `RegisterPassable` return type — but the trailing `()`
            # CALLS the C function with no arguments and types its return value
            # as a function pointer, which the next call then jumps into. See
            # `tests/cuda/test_dlhandle_get_function_arity.mojo`.
            #
            # Get Mojo's internal stream
            var get_stream = self._lib.get_function[_CUptr]("intercept_get_mojo_stream")
            self._mojo_stream = get_stream()
            if stream_of_context and Int(self._mojo_stream) != 0:
                # The interceptor is hooked (it has seen a launch), so its
                # driver entry points work; its stream may be another context's.
                var own = CUDA(ctx.stream())
                if not own:
                    raise Error("[CUDAGraph] MAX returned no stream for this context")
                self._mojo_stream = _CUptr(unsafe_from_address=Int(own.value()))

            if Int(self._mojo_stream) == 0:
                # A NULL stream has TWO causes and they need opposite fixes.
                # This message used to assert one of them ("MAX destroys its
                # stream after a synchronize") as though it were the only
                # possibility — a diagnosis this repo's own
                # `tests/cuda/probe_max_stream_lifetime.mojo` had already
                # REFUTED (MAX keeps one stream and tears it down when the
                # DeviceContext dies; see the `_ctx` field). Stating a stale
                # conclusion as fact sent a real debugging session hunting in
                # the wrong place, so the message now separates the cases
                # instead of picking one.
                #
                # The launch count is what separates them: the interceptor
                # records `g_mojo_stream` on EVERY `cuLaunchKernelEx`, so if
                # it has seen no launches at all it is not hooked into this
                # process and the NULL says nothing about MAX's streams.
                var get_launches = self._lib.get_function[c_int](
                    "intercept_get_launch_count"
                )
                var launches = Int(get_launches())
                self._state = 3  # DISABLED
                _print_capture_disabled(launches)
            else:
                # Create replay stream
                var stream_create = self._lib.get_function[c_int](
                    "intercept_stream_create"
                )
                var stream_alloc = alloc[_CUptr]({count = 1})
                var stream_buf = stream_alloc.unsafe_ptr()
                stream_buf.unsafe_write(_uninit[_CUptr]())
                _ = stream_create(stream_buf)
                self._replay_stream = stream_buf[]
                dealloc(stream_alloc^)
        else:
            self._lib = _uninit[OwnedDLHandle]()

    def begin_capture(mut self) raises:
        """Begin capturing GPU kernel launches. No-op on non-NVIDIA."""
        comptime if not has_nvidia_gpu_accelerator():
            return

        # ⚠ CHECK STATE FIRST. On the DISABLED path `_mojo_stream` was never
        # assigned a real value, so reading it below would be reading
        # uninitialized memory — and a non-zero garbage word would sail past
        # the check and hand a junk pointer to the driver.
        if self._state == 3:
            raise Error(
                "[CUDAGraph] disabled — capture is unavailable on this MAX."
                " Callers should check is_disabled() and run their work"
                " directly (maybe_capture_replay does)."
            )
        if Int(self._mojo_stream) == 0:
            raise Error(
                "[CUDAGraph] Mojo stream not discovered. "
                "Ensure at least one GPU kernel ran before init."
            )
        if self._state == 1:
            raise Error("[CUDAGraph] Already capturing.")

        # Destroy previous graph if re-capturing
        if self._state == 2:
            var exec_destroy = self._lib.get_function[c_int]("intercept_graph_exec_destroy")
            _ = exec_destroy(self._exec)
            var graph_destroy = self._lib.get_function[c_int]("intercept_graph_destroy")
            _ = graph_destroy(self._graph)
            self._exec = _uninit[_CUptr]()
            self._graph = _uninit[_CUptr]()

        # ⚠ A PRE-CAPTURE `intercept_stream_synchronize` USED TO BE HERE AND
        # WAS REMOVED — IT IS THE CALL THAT CRASHED. It was added to keep the
        # interceptor's sync-suppression honest (drain before the window so
        # answering "already idle" is true). The empty-capture probe then
        # faulted inside the driver ON THAT VERY CALL, with no capture active,
        # which is what proved the failure is not capture-specific: this shim
        # cannot safely call the driver with MAX's stream handle at all.
        # Do not reinstate it without fixing that first.
        var begin_capture = self._lib.get_function[c_int]("intercept_stream_begin_capture")
        var r = begin_capture(self._mojo_stream)
        if r != 0:
            raise Error("[CUDAGraph] cuStreamBeginCapture failed: " + String(r))
        self._state = 1

    def end_capture(mut self) raises:
        """End capture and instantiate the graph. No-op on non-NVIDIA."""
        comptime if not has_nvidia_gpu_accelerator():
            return

        if self._state != 1:
            raise Error("[CUDAGraph] Not capturing.")

        # End capture.
        #
        # Symbol resolution is hoisted ABOVE each allocation throughout this
        # method. `get_function` raises if the symbol is missing, and an
        # `Allocation` must be consumed on every path out of the scope — so a
        # raising call sandwiched between the allocation and its `dealloc`
        # leaks the buffer (and does not compile). Resolving first leaves no
        # raising call in the window, which is why none of these need a
        # `try`/`except`.
        var end_capture = self._lib.get_function[c_int](
            "intercept_stream_end_capture"
        )
        var graph_alloc = alloc[_CUptr]({count = 1})
        var graph_buf = graph_alloc.unsafe_ptr()
        graph_buf.unsafe_write(_uninit[_CUptr]())
        var r_end = end_capture(self._mojo_stream, graph_buf)
        self._graph = graph_buf[]
        dealloc(graph_alloc^)

        if r_end != 0:
            self._state = 0
            raise Error(
                "[CUDAGraph] cuStreamEndCapture failed: " + String(r_end)
            )

        # Count nodes
        var get_nodes = self._lib.get_function[c_int](
            "intercept_graph_get_nodes"
        )
        var num_alloc = alloc[UInt64]({count = 1})
        var num_buf = num_alloc.unsafe_ptr()
        num_buf.unsafe_write(UInt64(0))
        _ = get_nodes(self._graph, num_buf)
        self._num_nodes = Int(num_buf[])
        dealloc(num_alloc^)

        if self._num_nodes == 0:
            self._state = 0
            raise Error(
                "[CUDAGraph] Captured 0 nodes. "
                "Ensure LD_PRELOAD is set with libcuda_intercept.so."
            )

        # Instantiate
        var instantiate = self._lib.get_function[c_int](
            "intercept_graph_instantiate"
        )
        var exec_alloc = alloc[_CUptr]({count = 1})
        var exec_buf = exec_alloc.unsafe_ptr()
        exec_buf.unsafe_write(_uninit[_CUptr]())
        var r_inst = instantiate(exec_buf, self._graph)
        self._exec = exec_buf[]
        dealloc(exec_alloc^)

        if r_inst != 0:
            self._state = 0
            raise Error(
                "[CUDAGraph] cuGraphInstantiate failed: " + String(r_inst)
            )

        self._state = 2

    def replay(self) raises:
        """Replay the captured graph (launch + sync). No-op on non-NVIDIA."""
        comptime if not has_nvidia_gpu_accelerator():
            return

        if self._state != 2:
            raise Error("[CUDAGraph] No graph captured.")

        var launch = self._lib.get_function[c_int]("intercept_graph_launch")
        _ = launch(self._exec, self._replay_stream)

        var stream_sync = self._lib.get_function[c_int]("intercept_stream_synchronize")
        _ = stream_sync(self._replay_stream)

    def replay_async(self) raises:
        """Replay without sync on replay stream. Call sync() later. No-op on non-NVIDIA.
        """
        comptime if not has_nvidia_gpu_accelerator():
            return

        if self._state != 2:
            raise Error("[CUDAGraph] No graph captured.")

        var launch = self._lib.get_function[c_int]("intercept_graph_launch")
        _ = launch(self._exec, self._replay_stream)

    def replay_on_mojo_stream(self) raises:
        """Replay on Mojo's main stream (implicit ordering with other kernels).

        Unlike replay_async() which uses a separate replay stream,
        this launches the graph on the same stream used by ctx.enqueue_function.
        This ensures correct ordering: kernels enqueued before this call
        (e.g. select_actions_gpu) complete before the graph starts,
        and the graph completes before kernels enqueued after.

        No explicit sync needed between pre-graph and post-graph operations.
        No-op on non-NVIDIA.
        """
        comptime if not has_nvidia_gpu_accelerator():
            return

        if self._state != 2:
            raise Error("[CUDAGraph] No graph captured.")

        var launch = self._lib.get_function[c_int]("intercept_graph_launch")
        _ = launch(self._exec, self._mojo_stream)

    def sync(self) raises:
        """Synchronize the replay stream. No-op on non-NVIDIA."""
        comptime if not has_nvidia_gpu_accelerator():
            return

        var stream_sync = self._lib.get_function[c_int]("intercept_stream_synchronize")
        _ = stream_sync(self._replay_stream)

    def is_disabled(self) -> Bool:
        """No usable stream — the caller must run its work directly.

        True when construction found no live Mojo stream to capture. Callers
        must check this INSTEAD OF assuming capture is available; see the note
        in `__init__` for why this is the normal state under MAX 26.5.0rc2.
        """
        comptime if not has_nvidia_gpu_accelerator():
            return False
        return self._state == 3

    def is_captured(self) -> Bool:
        """Whether a graph is ready for replay."""
        comptime if not has_nvidia_gpu_accelerator():
            return False
        return self._state == 2

    def num_nodes(self) -> Int:
        """Number of kernel nodes in the captured graph."""
        return self._num_nodes


# ──────────────────────────────────────────────────────────────────────
# maybe_capture_replay — capture-lifecycle harness behind a closure.
# ──────────────────────────────────────────────────────────────────────


def maybe_capture_replay[
    STEP: def () capturing raises -> None,
    VERBOSE: Bool = True,
](mut graph: Optional[CUDAGraph], ctx: DeviceContext) raises:
    """Capture `STEP` into `graph` on first call; replay it thereafter.

    One generic helper owns the whole capture lifecycle (warmup → begin →
    capture → end → replay) behind a comptime *capturing closure*, so a
    training loop never inlines a `comptime if USE_CUDA_GRAPH` maze and the
    caller's trainer never learns about `CUDAGraph`.

    `STEP` is a comptime capturing closure that mutably captures the caller's
    state (e.g. the trainer + ctx) and enqueues the *pure device-kernel* step
    — NO host work. It must enqueue the SAME kernel sequence every call so the
    captured graph stays valid on replay (sampling included, so each replay
    draws a fresh minibatch via the device RNG counter).

    `graph` is the caller-owned capture slot (None until first capture). On
    NVIDIA: the first call runs `STEP` once to settle the stream, then
    captures a second run; later calls replay on the Mojo stream (implicit
    ordering with kernels enqueued before/after). On non-NVIDIA: `CUDAGraph`
    is a compile-time no-op, so this just runs `STEP()` every call —
    bit-identical to the non-captured path (the Apple-Silicon verification
    path; real capture/replay needs an NVIDIA run).

    Host bookkeeping (step counters, metric flush cadence) is intentionally
    NOT here — keep it in the caller's loop, advanced once per logical update,
    so it stays correct whether the step ran directly or via replay.

    `VERBOSE=False` skips the node-count line — for a caller that re-captures
    by design (a schedule baked into kernel arguments) and printed it once."""
    comptime if has_nvidia_gpu_accelerator():
        if not graph:
            STEP()
            ctx.synchronize()
            var g = CUDAGraph(ctx)

            # ⚠ NO USABLE STREAM -> RUN DIRECTLY, FOREVER, WITHOUT CRASHING.
            # `STEP()` has already run for this call, so returning here is
            # correct, not a skipped update. The disabled graph is STORED so
            # later calls take the `else` branch below instead of retrying
            # construction (which would re-`synchronize()` every single call
            # and re-print the warning).
            if g.is_disabled():
                graph = g^
                return

            g.begin_capture()
            STEP()
            g.end_capture()
            # Diagnostic: confirm the harness actually captured kernels (a
            # tiny/zero node count means capture silently failed — e.g. the
            # closure enqueued on a different stream than the one being
            # captured). Printed once per graph (first call only).
            comptime if VERBOSE:
                print(
                    "[CUDA Graph] maybe_capture_replay captured",
                    g.num_nodes(),
                    "nodes",
                )
            graph = g^
        elif graph.value().is_disabled():
            # Capture was never possible; the closure is the whole step.
            STEP()
        else:
            graph.value().replay_on_mojo_stream()
    else:
        STEP()
