"""Does a split-K GEMM abort CUDA-graph capture?

    pixi run -e nvidia mojo run -I . \\
        benchmarks/bench_cuda_graph_splitk_capture.mojo

## Why this is the first thing to run

`docs/ACT_GPU_DATA_PATH.md` proposes moving ACT's data path onto the device and
then capturing the training step. Every prerequisite is either already done or
ours to fix — except one, and it is not ours.

MAX's split-K allocates its reduction workspace PER CALL:

    // matmul/gpu/__init__.mojo:1845
    if runtime_config.num_k_partitions > 1:
        var work_space_data = ctx.enqueue_create_buffer[work_space_type](
            runtime_config.num_k_partitions * M * N
        )

and in the nsys trace those land as `cuMemAlloc_v2` — the SYNCHRONOUS driver
allocator, not `cuMemAllocAsync`. A synchronous allocation inside a capture
region is illegal on CUDA and aborts the capture. ACT hits
`multistage_gemm_split_k` about **46 times per step** (its dW GEMMs,
`grad_w = xᵀ @ go`, where K is `batch x tokens`), so if that reasoning holds,
CUDA graphs for ACT are blocked until Modular reuses that workspace.

It also explains why SAC captures fine today: its MLP GEMMs are small and
square-ish, so `select_config` never partitions K.

⚠ **That is a chain of inference, not a measurement.** MAX may special-case
capture, or its `_DeviceBufferMode._ASYNC` buffers may reach a stream-ordered
pool that nsys reports differently. This file settles it in one run instead of
building a plan on top of a guess.

## What it does

Three capture attempts through the repo's own `maybe_capture_replay`, so the
lifecycle is exactly the one the training drivers use:

  A. a NON-split-K GEMM  — the control. If this fails, the harness is wrong,
     not the hypothesis.
  B. a split-K GEMM      — `[256 x 2592] @ [2592 x 256]`, confirmed to pick
     `multistage_gemm_split_k` in `bench_matmul_alloc_splitk_and_bmm.mojo`.
  C. B again, after a warm-up call outside capture — in case the allocation is
     a one-off that a settled stream avoids.

## ANSWERED, RTX 5090, MAX 26.5.0: BLOCKED

    A. control, plain multistage   cuStreamBeginCapture rc=0
                                   cuStreamEndCapture   rc=0
                                   captured 1 nodes, checksum ok   -> CAPTURES

    B. split-K, cold               cuStreamBeginCapture rc=0, then
      oom log request=512KB reason="graph capturing in progress, no driver fallback"
      CAPTURE FAILED: Memory manager cannot satisfy allocation
                      (memory manager is disabled): cuda[0] on graphFreeList

**MAX's allocator refuses to serve an allocation while a graph is capturing**
— "no driver fallback" — and the graph-specific pool it would serve it from,
`graphFreeList`, is EMPTY. So the split-K reduction workspace cannot be
allocated inside a capture region, and any step containing a split-K GEMM
cannot be captured. ACT's backward has ~46 of them per step.

Two things worth carrying to Modular beyond the allocation itself:

  * The existence of `graphFreeList` says MAX already has the CONCEPT of a
    capture-time pool; it is just never pre-reserved. If it could be warmed
    before capture (or sized by a knob), this would be unblocked without any
    change to split-K itself.
  * The failure is not clean. After the allocation fails, the error path calls
    `cuStreamSynchronize` ON THE CAPTURED STREAM — illegal in every capture
    mode — and the process aborts instead of surfacing a catchable error. The
    intercept shim prints it: `!! cuStreamSynchronize DURING CAPTURE`.

## Reading it

    A ok, B ok, C ok      -> NOT blocked. Capture allocates fine (or MAX pools
                             it). Proceed with the plan in ACT_GPU_DATA_PATH.
    A ok, B fails         -> BLOCKED, as reasoned. The Modular report stops
                             being a 22% tax and becomes the gate on a whole
                             optimization; say so when filing.
    A ok, B fails, C ok   -> the allocation is first-call only. Warm every
                             shape before capture and the plan survives.
    A fails               -> harness problem. Ignore B and C.

⚠ A "pass" here means capture COMPLETED, not that replay is correct. If B
passes, the next question is whether the replayed graph reuses a workspace that
was freed after capture — which would be a silent wrong-answer bug, not a
crash. Every arm therefore checks the output against its ANALYTIC value
(`K * 0.01 * 0.02` per element) and reports a mismatch as "the step did not
run", because a capture that silently computes nothing would otherwise look
exactly like a pass. The first draft of this file did precisely that on an M1.

⚠ On NVIDIA, `maybe_capture_replay` already runs `STEP()` once and
synchronizes BEFORE it captures a second run — so all three arms get a warm-up
there, and C differs only in having also allocated and freed the workspace
three times first. C stays because "the allocation is a first-call artifact"
and "the allocation is fine under capture" are different answers.

⚠ TWO drafts of this file crashed on the 5090 before it ran, both from the same
rule, and `noeira/cuda/graph.mojo:54` states that rule in full: **Mojo
destroys a value at its LAST USE, and MAX's `DeviceContext` destructor
SYNCHRONIZES AND DESTROYS the stream.** `AsyncRT_DeviceContext_release` was
frame #12 of both stack traces, exactly as that comment predicts.

    draft 1  output allocated as a `Tensor`, view built from `c.dev.value()`
             (a temporary unwrap) -> buffer dropped at the `TileTensor`
             construction -> CUDA_ERROR_MISALIGNED_ADDRESS at begin_capture
    draft 2  buffers fixed, but the closure still mentioned `ctx` directly, so
             it took its own copy, that copy died, and the stream being
             captured died with it -> CUDA_ERROR_ILLEGAL_ADDRESS

Both faults surfaced at `begin_capture()` because CUDA errors are deferred; the
actual fault was the warm-up `STEP()` immediately before. On Apple the first
bug read instead as arms A and B silently computing 0.0, which I wrote off as
an unexplained quirk of the no-op path. It was not.

The fix is the shape SAC already uses: ONE struct owns the context and the
buffers, and the closure mentions only that struct and calls a method on it
(`trainer.train_device_kernels()`,
`driver_offpolicy_discrete.mojo:867`). Copy it, do not improvise.

⚠ A harness whose CONTROL arm fails is not evidence about split-K. It is
evidence about the harness.
"""

from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major
from linalg.matmul import matmul as max_matmul

from std.sys import default_accelerator

from noeira.cuda import CUDAGraph, maybe_capture_replay

comptime DT = DType.float32

# `[256 x 2592] @ [2592 x 256]` picks multistage_gemm_split_k on a 5090;
# `[2592 x 256] @ [256 x 256]` picks plain multistage. Both PASS the dispatch
# gate (n % 128 == 0, k % 32 == 0, k >= 128), so neither can reach the vendor
# fallback — the only difference is whether K gets partitioned.
comptime SK_M = 256
comptime SK_K = 2592
comptime SK_N = 256

comptime OK_M = 2592
comptime OK_K = 256
comptime OK_N = 256

comptime REPLAYS = 4

# a = 0.01 everywhere, b = 0.02 everywhere, so every output element is
# K * 0.0002 and the first-8 checksum is analytic. ⚠ Checking it is what makes
# a "pass" mean something: `maybe_capture_replay` is a compile-time no-op off
# NVIDIA, and the first draft of this file reported "NOT BLOCKED" on an M1
# where the closure had written nothing at all. A verdict from a run that
# computed the wrong number is not a verdict.
comptime AVAL = 0.01
comptime BVAL = 0.02


def _checksum(
    cbuf: DeviceBuffer[DT], ctx: DeviceContext, total: Int, n: Int
) raises -> Float64:
    """Sum of the first `n` outputs — the guard against a capture that
    SUCCEEDS and then replays against a workspace that no longer exists.

    Takes the DeviceBuffer directly rather than a `Tensor`, so the buffer the
    GEMM wrote is provably the buffer being read; see `_attempt`."""
    var host = ctx.enqueue_create_host_buffer[DT](total)
    ctx.enqueue_copy(host, cbuf)
    ctx.synchronize()
    var s = Float64(0.0)
    for i in range(n):
        s += Float64(host[i])
    return s


struct _Probe[M_: Int, K_: Int, N_: Int](Movable):
    """Owns the context AND the buffers, and does the GEMM in a METHOD.

    ⚠⚠ THIS SHAPE IS LOAD-BEARING, and two rounds of crashes came from not
    using it. `noeira/cuda/graph.mojo:54` spells out why: Mojo destroys a
    value at its LAST USE, and MAX's `DeviceContext` destructor SYNCHRONIZES
    AND DESTROYS the stream. A `capturing` closure that mentions `ctx`
    directly takes its own copy, that copy dies, and the stream being captured
    goes with it — `AsyncRT_DeviceContext_release` was frame #12 of both
    faults, exactly as that comment predicts. Earlier drafts also let the
    output DeviceBuffer die at the `TileTensor` construction.

    So the closure must mention ONE thing that owns everything and call a
    method on it, which is precisely what SAC does:
    `trainer.train_device_kernels()` in
    `driver_offpolicy_discrete.mojo:867`. Copy that, do not improvise."""

    var ctx: DeviceContext
    var a: DeviceBuffer[DT]
    var b: DeviceBuffer[DT]
    var c: DeviceBuffer[DT]

    def __init__(out self, ctx: DeviceContext) raises:
        self.ctx = ctx
        self.a = ctx.enqueue_create_buffer[DT](Self.M_ * Self.K_)
        self.b = ctx.enqueue_create_buffer[DT](Self.K_ * Self.N_)
        self.c = ctx.enqueue_create_buffer[DT](Self.M_ * Self.N_)
        self.a.enqueue_fill(Scalar[DT](AVAL))
        self.b.enqueue_fill(Scalar[DT](BVAL))
        self.c.enqueue_fill(Scalar[DT](0))
        ctx.synchronize()

    def __init__(out self, *, deinit move: Self):
        self.ctx = move.ctx^
        self.a = move.a^
        self.b = move.b^
        self.c = move.c^

    def step(mut self) raises:
        """The whole captured sequence: one GEMM, no host work."""
        var av = TileTensor(self.a, row_major[Self.M_, Self.K_]())
        var bv = TileTensor(self.b, row_major[Self.K_, Self.N_]())
        var cv = TileTensor(self.c, row_major[Self.M_, Self.N_]())
        max_matmul[target="gpu"](cv, av, bv, self.ctx)

    def wipe(mut self) raises:
        self.c.enqueue_fill(Scalar[DT](0))
        self.ctx.synchronize()

    def checksum(mut self, n: Int) raises -> Float64:
        """Sum of the first `n` outputs. A capture that succeeds and then
        replays against a workspace freed after capture is a silent wrong
        answer, not a crash — so every arm checks its ANALYTIC value."""
        var host = self.ctx.enqueue_create_host_buffer[DT](Self.M_ * Self.N_)
        self.ctx.enqueue_copy(host, self.c)
        self.ctx.synchronize()
        var s = Float64(0.0)
        for i in range(n):
            s += Float64(host[i])
        return s


def _attempt[
    M_: Int, K_: Int, N_: Int
](ctx: DeviceContext, label: String, warm: Bool) raises -> Bool:
    var p = _Probe[M_, K_, N_](ctx)

    if warm:
        # Settle the stream and let any first-call allocation happen OUTSIDE
        # the capture region, then WIPE — so the warm-up's correct output
        # cannot stand in for a capture path that computed nothing.
        for _ in range(3):
            p.step()
        p.ctx.synchronize()
        p.wipe()

    var graph: Optional[CUDAGraph] = None

    # ⚠ The closure mentions `p` and NOTHING else. See `_Probe`.
    def _step() capturing raises -> None:
        p.step()

    print("  " + label + " ...")
    try:
        for _ in range(REPLAYS):
            maybe_capture_replay[_step](graph, p.ctx)
        p.ctx.synchronize()
    except e:
        print("    CAPTURE FAILED: " + String(e))
        return False
    var s = p.checksum(8)
    var want = Float64(K_) * AVAL * BVAL * 8.0
    # ⚠ RELATIVE, and loose enough for TF32. A 5090 runs fp32 matmul through
    # TF32 tensor cores by default — ~10 mantissa bits — and arm A came back
    # 0.40914977 against 0.4096, a relative error of 1.1e-3 that a 1e-3 gate
    # called "the step did not run". The repo already learned this once
    # (a1efd4f2, "CPU/GPU parity must be relative and TF32-aware"); the point
    # of this check is to catch a step that computed NOTHING, which is off by
    # 100%, not to police the last bits.
    var ok = abs(s - want) <= 5e-3 * want
    print("    captured + replayed x" + String(REPLAYS)
          + "   checksum(first 8) = " + String(s)
          + "   expected " + String(want)
          + ("   ok" if ok else "   <== WRONG, the step did not run"))
    return ok


def main() raises:
    var ctx = DeviceContext()
    print("=" * 74)
    print("Does a split-K GEMM abort CUDA-graph capture? —", ctx.name())
    print("=" * 74)
    comptime if not default_accelerator().is_nvidia_gpu():
        print("  ⚠ NOT NVIDIA. `maybe_capture_replay` is a compile-time no-op")
        print("    here, so nothing is captured and NO VERDICT IS POSSIBLE.")
        print("    The arms below only check that the harness computes the")
        print("    right numbers. Run this on the 5090 for the answer.")
        print("")

    var a_ok = _attempt[OK_M, OK_K, OK_N](
        ctx, "A. control, plain multistage (no split-K, no workspace)", False
    )
    var b_ok = _attempt[SK_M, SK_K, SK_N](
        ctx, "B. split-K, cold (workspace allocated inside capture?)", False
    )
    var c_ok = _attempt[SK_M, SK_K, SK_N](
        ctx, "C. split-K, warmed outside the capture region first", True
    )

    print("=" * 74)
    comptime if not default_accelerator().is_nvidia_gpu():
        print("  NO VERDICT — not an NVIDIA device (see above).")
        if a_ok and b_ok and c_ok:
            print("  The harness itself computes the expected values.")
        else:
            print("  ⚠ and the HARNESS is wrong: an arm produced the wrong")
            print("    checksum, so fix that before trusting a 5090 run.")
        return
    if not a_ok:
        print("  A FAILED -> harness problem, B and C say nothing.")
    elif b_ok and c_ok:
        print("  NOT BLOCKED. Capture tolerates the split-K workspace.")
        print("  -> proceed with docs/ACT_GPU_DATA_PATH.md as written.")
        print("  ⚠ verify the checksums above are IDENTICAL across the three")
        print("    runs; a capture that succeeds but replays against a freed")
        print("    workspace is a silent wrong answer, not a crash.")
    elif c_ok:
        print("  FIRST-CALL ONLY. Warm every GEMM shape before capture and")
        print("  the plan survives; add that to the capture preamble.")
    else:
        print("  BLOCKED, as reasoned. ACT cannot be captured while its dW")
        print("  GEMMs allocate a split-K workspace per call (~46/step).")
        print("  -> this raises docs/MODULAR_MATMUL_ALLOC_REPORT.md from")
        print("     'a 22% tax' to 'the gate on CUDA graphs'. Say so when")
        print("     filing, and mention that SAC captures fine only because")
        print("     its MLP shapes never partition K.")
    print("=" * 74)
