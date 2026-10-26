"""Does MAX's own `DeviceGraph` work here — and can it replace our LD_PRELOAD shim?

    pixi run -e nvidia mojo run -I . \\
        benchmarks/bench_device_graph_spike.mojo

    # the point of arm 0: NO interceptor, NO activation, NO LD_PRELOAD.
    pixi run -e nvidia mojo build -I . -o /tmp/dg_spike \\
        benchmarks/bench_device_graph_spike.mojo
    /tmp/dg_spike                      # <- run it DIRECTLY, bare shell

    # skip the arm that may kill the process (see arm D):
    NOEIRA_SPIKE_SPLITK=0 pixi run -e nvidia mojo run -I . \\
        benchmarks/bench_device_graph_spike.mojo

## Why this exists

`docs/MODULAR_SOURCE_DEEP_DIVE.md` found that MAX 26.5 SHIPS the API our
`noeira/cuda/graph.mojo` reimplements by hand:

    from max.gpu.host import DeviceGraph, DeviceGraphBuilder

1616 lines in `max.mojoc`, exported from `max/gpu/host/__init__.mojo`, with
MAX's own test at
`references/modular-max-v26.5/max/kernels/test/gpu/device_context/test_device_graph_builder.mojo`
(this file's syntax is copied from it, deliberately).

What we have instead is ~400 lines of `LD_PRELOAD` interceptor that discovers
MAX's INTERNAL stream by watching `cuLaunchKernelEx` and then calls
`cuStreamBeginCapture` on a stream we do not own. That file's own comments are
the bill for it: `cuStreamDestroy` on our handle, driver SIGSEGVs in
BeginCapture / EndCapture / GetCaptureInfo, a load-bearing `_ctx` field, and a
failure mode where capture "succeeds" with ZERO nodes and nothing says so. Plus
[[_the_preload_is_a_property_of_the_process_not_the_binary]]: a binary you
BUILD and then run directly silently loses capture entirely.

`DeviceGraph` is a different mechanism — explicit DAG construction, not stream
capture — so none of that applies. It is a supported API, it needs no preload,
and `recording_context()` claims that ordinary `DeviceContext` code records
unmodified, which would mean our existing `STEP()` closures port as-is.

⚠ THAT IS A DOCSTRING, NOT A MEASUREMENT. This file is the measurement.

## The arms, in order, each independently reportable

  0. is it even here?      imports resolve, `DeviceContext` alive, device named.
  A. add_function          the explicit-DAG path: one node, build must NOT
                           execute, replay must execute exactly once.
  B. recording_context     OUR kernel style (LayoutTensor args) through the
                           ordinary `enqueue_function` API, four launches, order
                           preserved. If this passes, a `STEP()` closure that
                           takes a `DeviceContext` records with NO shim.
  C. a MAX library GEMM    `max_matmul` recorded through `recording_context()`.
                           This is the arm that decides whether our real
                           training steps port, because they are mostly library
                           GEMMs, not hand kernels.
  D. a SPLIT-K GEMM        the known blocker. Predicted to FAIL, and to fail
                           SILENTLY — see below.
  E. timing                eager enqueue vs `graph.replay()`, us/iter.

## What arm D is really testing, and why it is last

Stream capture cannot contain a split-K GEMM at all: MAX's allocator refuses to
serve `enqueue_create_buffer` while a graph is capturing and ABORTS the process
(`benchmarks/bench_cuda_graph_splitk_capture.mojo`;
[[_split_k_cannot_be_cuda_graph_captured]]). `DeviceGraph` has a documented
capture-scoped pool for exactly this — `builder.create_buffer`, "all allocations
created for a device graph should go through this method". But `multistage_gemm`
calls `ctx.enqueue_create_buffer`, and `recording_context()`'s docstring says
buffer allocation FORWARDS TO THE BACKING CONTEXT.

So the predicted failure is worse than an abort: the workspace is allocated
EAGERLY at record time, `multistage_gemm` frees it at `_ = work_space_data^`
when the call returns, and the recorded node keeps the raw pointer. That is a
dangling device pointer on replay — **a wrong answer, not a crash**, and quite
possibly a RIGHT answer on the first replay while the freed block is still
untouched in the pool.

Hence arm D replays TWICE with an allocator CHURN in between, and checks an
analytic checksum both times. A pass on replay 1 and a fail on replay 2 is the
signature of the dangling workspace and is the most likely outcome.

⚠ It may also fault or abort. It runs LAST so that A–C have already printed,
and `NOEIRA_SPIKE_SPLITK=0` skips it.

⚠ AND IT MUST PROVE IT IS SPLIT-K ON *THIS* CARD. `[256 x 2592] @ [2592 x 256]`
partitions K on a 5090; `select_config` hardcodes `A100.sm_count = 108` in its
wave estimate, so another part can decide otherwise and the arm would be
measuring nothing. It prints `num_k_partitions` and says so when it is 1.
([[_a_capture_abort_names_only_the_first_blocker]]: a size is not an identity.)

## ANSWERED, RTX 5090, MAX 26.5.0 — A, B, C, E PASS; **D FAILS**

    A. add_function        build->0.0   replay1->7.0    replay2(after wipe)->7.0
    B. recording_context   build->0.0   replay1->12.0   replay2(from 99)->12.0
    C. max_matmul          build->0.0   replay1->0.40914977        want 0.4096
    D. SPLIT-K (P=2)       pool reissues a freed 512KB block: YES
                           replay1->4.14263630  replay2(freed churn)->4.14263630
                           replay3(LIVE poison)->4.14263630   want 4.14720
                           **live buffers written through: 1/16**   <-- FAIL
    E. 32 launches         eager 100.62 -> replay 28.57 us/iter = 3.52x
                           (3.14 -> 0.89 us/launch)
                           counter->13440.0 = 13440.0, so they really ran

**A and B settle the migration.** DeviceGraph records both MAX's kernel style
and OURS, with no interceptor, and `build` executes nothing — so a `STEP()`
closure ports by taking a `DeviceContext` ARGUMENT instead of capturing one.
That is the entire change `noeira/cuda/graph.mojo` needs to become deletable,
and it is independent of everything below.

## D: the prediction was right, and the failure is WORSE than an abort

The dispatch question is settled — `-D LOGGING_LEVEL=INFO` prints
`K partitions:` **1, 1, 2, 2**: exactly four lines for the program's four
`matmul` calls (arm C warm + recorded, arm D warm + recorded), in order, and
the replays add none, because a replay re-runs recorded NODES rather than the
host dispatch. The recorded arm-D call took the split-K branch at P=2.

And with a live-aliasing probe in place it FAILS: **`live buffers written
through: 1/16`.** One 512KB buffer, allocated after the graph was built and
still held by us, was written through while the graph replayed. Exactly one —
one workspace-sized block, one claimant.

So `multistage_gemm` does what its source says: allocates the reduction
workspace with `ctx.enqueue_create_buffer`, which `recording_context` FORWARDS
to the backing context, and frees it at `_ = work_space_data^` while the
recorded node keeps the raw pointer. **The GEMM's own answer stays right
(4.14263630 on all three replays) because the kernel writes that workspace
before `split_k_reduce` reads it, inside one graph, in order. What it destroys
is whatever the allocator handed the block to next.**

That is strictly worse than stream capture, which aborts the process on this
shape ([[_split_k_cannot_be_cuda_graph_captured]]). Here nothing raises,
nothing crashes, the checksum is correct, and an unrelated tensor is silently
overwritten. **ACT issues ~46 split-K GEMMs per step; recording one would put
46 stray writers into live memory and the loss would just be wrong.**

## The three vacuity holes this arm went through, because they are the lesson

I predicted this failure in the first draft, then MEASURED A PASS TWICE, and
both passes were my own test being blind. Each fix found the next hole:

  1. **the churn could not reach the block** — until `_pool_recycles` proved
     the allocator reissues a freed 512KB block (`YES`);
  2. **the recorded call might never have split K** — `select_config` is a
     CHOOSER, not the dispatch gate
     ([[_select_config_is_not_the_dispatch_gate]]), and both branches print
     the same number. Only the kernel log separated them;
  3. **the poison was FREED before the replay** — so nothing owned that memory,
     and the split-K kernel simply overwrote it. A dangling-but-mapped pointer
     yields the RIGHT ANSWER. Stage 2 could only ever catch an UNMAP, which
     MAX's pool never does.

⚠ **GENERAL SHAPE, worth more than this arm:** a scratch buffer that is
written-then-read INSIDE one graph cannot be caught by poisoning it beforehand.
Poison something that OUTLIVES the replay and check it afterwards. And the
probe prints the VALUE it found beside the expected partial sum, because a
count is not an identity — "something wrote here" has several authors,
`K/P * a * b` has one.

⚠ **E is a launch-overhead CEILING, not a step.** 32 trivial kernels: 3.14 ->
0.89 us per launch, and the replay figure carries one synchronize for the whole
chain. A real step's kernels do work that overlaps the launch
([[_a_per_call_sweep_is_an_upper_bound_on_a_step]]: a per-call sweep predicted
+0.57ms and the step moved +0.06ms, 10x over).

⚠ **Arm E CRASHED on the first 5090 run, and it was this file's bug.**
`CUDA_ERROR_ILLEGAL_ADDRESS` at `AsyncRT_DeviceContext_release`: `buf`'s last
mention was the `LayoutTensor` construction, so Mojo freed the DeviceBuffer
there and every launch wrote through a dangling view. Arms A and B were saved
only by happening to read `buf` back at the end — **the bug is invisible in an
arm that checks its value.** Fixed by a closing read-back that also asserts the
launch arithmetic.

⚠ **The runs do NOT prove the "no LD_PRELOAD" claim.** `[intercept] Mojo
stream: 0x...` appears in their output: `pixi run -e nvidia` applies the nvidia
activation, which preloads `libcuda_intercept.so` whether or not anything uses
it ([[_the_preload_is_a_property_of_the_process_not_the_binary]]). This file
never calls it; the built-binary line at the top is what SHOWS that.

## Reading the result

    0,A,B pass                 DeviceGraph is live here. The shim is replaceable
                               for hand-written kernel sequences at minimum.
    C passes                   library calls record. Port `maybe_capture_replay`
                               onto `DeviceGraph` — the closure signature has to
                               grow a `DeviceContext` argument (today it captures
                               one), which is the whole migration.
    C fails                    the interesting failure. Report WHICH call raised;
                               a `synchronize()` inside a library path would do
                               it, and `recording_context()` raises on host waits
                               by design.
    D fails (as predicted)     unchanged conclusion: own the split-K workspace.
                               DeviceGraph does not rescue split-K by itself, it
                               just fails differently (silently, not loudly).
    D passes                   then MAX routes that allocation somewhere better
                               than its docstring implies — re-check
                               `_split_k_cannot_be_cuda_graph_captured`.

⚠ Off NVIDIA this file compiles and runs but there is NO VERDICT: Metal supports
neither stream capture nor device graphs ([[_the_metal_launch_floor_is_command_
buffer_retirement]]). Every arm is expected to RAISE on Apple, and the arm
harness reports that as "unsupported here", not as a failure. Run it on the
5090 for an answer.
"""

from std.math import ceildiv
from std.os import getenv
from std.sys import default_accelerator
from std.time import perf_counter_ns

from layout import TileTensor, row_major
from noeira.core.tiles import Tile, tile
from max.gpu.host import (
    DeviceBuffer,
    DeviceContext,
    DeviceGraph,
    DeviceGraphBuilder,
)
from linalg.matmul import matmul as max_matmul
from linalg.utils_gpu import select_config

comptime DT = DType.float32

comptime N = 1024
comptime BLOCK = 256
comptime GRID = ceildiv(N, BLOCK)

# Same shapes as `bench_cuda_graph_splitk_capture.mojo`, so the two files are
# directly comparable. Both PASS the dispatch gate (n % 128 == 0, k % 32 == 0,
# k >= 128) — neither can reach the vendor fallback, so the ONLY difference
# between them is whether `select_config` partitions K.
comptime OK_M = 2592  # plain multistage
comptime OK_K = 256
comptime OK_N = 256

comptime SK_M = 256  # multistage_gemm_split_k on a 5090
comptime SK_K = 2592
comptime SK_N = 256

# Fails `multi_gemm_cond` (`m>1 and n%128==0 and k%32==0 and k>=128`,
# matmul/gpu/__init__.mojo:591) on the K test — k=64 < 128 — so it routes to the
# VENDOR BLAS path, which allocates 32MB per call at blas.mojo:780. That is the
# shape the ACT conv GEMMs land on (k = OC = 64).
comptime VD_M = 256
comptime VD_K = 64
comptime VD_N = 256
comptime VD_PROBE_ELEMS = 32 * 1024 * 1024 // 4  # match the 32MB vendor alloc

comptime AVAL = 0.01
comptime BVAL = 0.02

comptime CHAIN = 32  # launches per iteration in the timing arm
comptime WARM = 10
comptime TIMED_ITERS = 200


# ─────────────────────────────────────────────────────────────────────────────
# Kernels.
#
# ⚠ TWO STYLES ON PURPOSE. `_fill_ptr` takes raw `UnsafePointer`s because that
# is what MAX's own device-graph test uses and arm A must not be the thing that
# fails. `_fill_lt` / `_bump_lt` take `LayoutTensor`, which is what EVERY kernel
# in this repo takes — arm B exists to prove that our style records too, and not
# only MAX's.
# ─────────────────────────────────────────────────────────────────────────────

from max.gpu import global_idx


def _fill_ptr(
    buf: Pointer[Float32, MutAnyOrigin], val: Int32, length: Int32
):
    var n = Int(length)
    var tid = global_idx.x
    if tid >= n:
        return
    buf[unsafe_offset=tid] = Float32(Int(val))


def _fill_lt(
    buf: Tile[DT, N], val: Int32
):
    var tid = Int(global_idx.x)
    if tid >= N:
        return
    buf[tid] = Scalar[DT](Int(val))


def _bump_lt(
    buf: Tile[DT, N], delta: Int32
):
    var tid = Int(global_idx.x)
    if tid >= N:
        return
    buf[tid] = buf[tid] + Scalar[DT](Int(delta))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _read_first(ctx: DeviceContext, buf: DeviceBuffer[DT]) raises -> Float64:
    """First element, host-side. The arms assert on an ANALYTIC value.

    ⚠ Reading one element is enough ONLY because every kernel here writes the
    whole buffer uniformly. Do not copy this into a test of a real kernel.
    """
    var host = ctx.enqueue_create_host_buffer[DT](1)
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()
    return Float64(host[0])


def _sum_first(
    ctx: DeviceContext, buf: DeviceBuffer[DT], total: Int, n: Int
) raises -> Float64:
    var host = ctx.enqueue_create_host_buffer[DT](total)
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()
    var s = Float64(0.0)
    for i in range(n):
        s += Float64(host[i])
    return s


def _close(got: Float64, want: Float64) -> Bool:
    """Relative, and loose enough for TF32.

    A 5090 runs fp32 matmul through TF32 tensor cores by default (~10 mantissa
    bits); `bench_cuda_graph_splitk_capture.mojo` saw 0.40914977 against 0.4096
    and a 1e-3 gate called that "the step did not run". The job of this check is
    to catch a step that computed NOTHING or read freed memory — both off by
    ~100% — not to police the last bits.
    """
    if want == 0.0:
        return abs(got) <= 1e-9
    return abs(got - want) <= 5e-3 * abs(want)


def _fixed2(x: Float64) -> String:
    """Two decimals. `String(Float64)` prints a 16-digit tail that has no
    business in a timing line and, on the 5090, overran the next print."""
    var scaled = Int(x * 100.0 + 0.5)
    var frac = scaled % 100
    var fs = String(frac) if frac >= 10 else "0" + String(frac)
    return String(scaled // 100) + "." + fs


def _verdict(label: String, ok: Bool, detail: String):
    print("    " + ("PASS  " if ok else "FAIL  ") + label + "   " + detail)


# ─────────────────────────────────────────────────────────────────────────────
# Arm A — the explicit-DAG path
# ─────────────────────────────────────────────────────────────────────────────


def arm_a(ctx: DeviceContext) raises -> Bool:
    """`add_function`: build must NOT execute; each replay must execute once.

    ⚠ "BUILD MUST NOT EXECUTE" IS HALF THE ASSERTION, and it is the half a
    careless harness drops. `noeira/cuda/graph.mojo`'s minimal test learned
    this: a graph layer that silently ran the work eagerly and replayed NOTHING
    passes any check that only looks at the final value.
    """
    print("  A. add_function — explicit node, no stream capture")

    var buf = ctx.enqueue_create_buffer[DT](N)
    buf.enqueue_fill(Scalar[DT](0))
    ctx.synchronize()

    var fill = ctx.compile_function[_fill_ptr]()

    # ⚠ `{imm}` is the capture list, and this spelling is copied verbatim from
    # MAX's own test. `DeviceGraph.create` takes
    # `def[o: ImmOrigin](mut DeviceGraphBuilder[o]) raises` — the builder's
    # origin is scoped to the call, so a node handle cannot escape it.
    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        _ = builder.add_function(
            fill,
            buf,
            Int32(7),
            Int32(N),
            grid_dim=GRID,
            block_dim=BLOCK,
        )

    var graph = DeviceGraph.create(ctx, build)
    ctx.synchronize()
    var after_build = _read_first(ctx, buf)

    graph.replay()
    ctx.synchronize()
    var after_1 = _read_first(ctx, buf)

    # Zero it so the second replay has to recompute rather than inherit.
    buf.enqueue_fill(Scalar[DT](0))
    ctx.synchronize()
    graph.replay()
    ctx.synchronize()
    var after_2 = _read_first(ctx, buf)

    var ok = _close(after_build, 0.0) and _close(after_1, 7.0) and _close(
        after_2, 7.0
    )
    _verdict(
        "one-node graph",
        ok,
        "build->"
        + String(after_build)
        + " (want 0)  replay1->"
        + String(after_1)
        + "  replay2(after wipe)->"
        + String(after_2)
        + " (want 7)",
    )
    if not ok and not _close(after_build, 0.0):
        print("      ^ the BUILD executed. That is eager enqueue wearing a")
        print("        graph's name, and every timing number below is void.")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Arm B — recording_context with OUR kernel style
# ─────────────────────────────────────────────────────────────────────────────


def arm_b(ctx: DeviceContext) raises -> Bool:
    """Four LayoutTensor launches through the ordinary `enqueue_function` API.

    THE ARM THAT DECIDES THE MIGRATION SHAPE. If unmodified `DeviceContext`
    code records, then `maybe_capture_replay`'s `STEP()` closure records too —
    it only has to take the context as an ARGUMENT instead of capturing one.

    ⚠ THE ARITHMETIC IS THE ASSERTION, not just the total. fill(5) then +1, +2,
    +4 lands on 12 only if all four recorded AND stayed in order. A graph that
    dropped the fill, or reordered the bumps against it, lands elsewhere — and
    on CUDA that ordering is a recorded dependency EDGE, not an implicit stream
    FIFO, so it is a real thing to check.
    """
    print("  B. recording_context — our LayoutTensor kernels, unmodified")

    var buf = ctx.enqueue_create_buffer[DT](N)
    buf.enqueue_fill(Scalar[DT](0))
    ctx.synchronize()
    var lt = tile[N](buf)

    var fill = ctx.compile_function[_fill_lt]()
    var bump = ctx.compile_function[_bump_lt]()

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        with builder.recording_context() as rec:
            rec.enqueue_function(
                fill, lt, Int32(5), grid_dim=GRID, block_dim=BLOCK
            )
            rec.enqueue_function(
                bump, lt, Int32(1), grid_dim=GRID, block_dim=BLOCK
            )
            rec.enqueue_function(
                bump, lt, Int32(2), grid_dim=GRID, block_dim=BLOCK
            )
            rec.enqueue_function(
                bump, lt, Int32(4), grid_dim=GRID, block_dim=BLOCK
            )

    var graph = DeviceGraph.create(ctx, build)
    ctx.synchronize()
    var after_build = _read_first(ctx, buf)

    graph.replay()
    ctx.synchronize()
    var after_1 = _read_first(ctx, buf)

    buf.enqueue_fill(Scalar[DT](99))
    ctx.synchronize()
    graph.replay()
    ctx.synchronize()
    var after_2 = _read_first(ctx, buf)

    var ok = _close(after_build, 0.0) and _close(after_1, 12.0) and _close(
        after_2, 12.0
    )
    _verdict(
        "4-launch chain",
        ok,
        "build->"
        + String(after_build)
        + " (want 0)  replay1->"
        + String(after_1)
        + "  replay2(from 99)->"
        + String(after_2)
        + " (want 12)",
    )
    if not ok and _close(after_2, 111.0):
        print("      ^ 99+1+2+4: the FILL did not record, the bumps did.")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Arms C / D — a MAX library GEMM through recording_context
# ─────────────────────────────────────────────────────────────────────────────


struct _Gemm[M_: Int, K_: Int, N_: Int](Movable):
    """Owns the context AND the operands.

    ⚠⚠ LOAD-BEARING, and two rounds of crashes in
    `bench_cuda_graph_splitk_capture.mojo` came from not doing it. Mojo
    destroys a value at its LAST USE, and MAX's `DeviceContext` destructor
    SYNCHRONIZES AND DESTROYS the stream. Separately, A GRAPH HOLDS RAW
    POINTERS: every operand must outlive the LAST REPLAY, not the build. Both
    rules point the same way — one struct owns everything, the closure mentions
    only that struct.
    """

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

    def gemm(mut self, ctx: DeviceContext) raises:
        """The recorded work. Takes the context as an ARGUMENT — that is the
        one change a `STEP()` closure needs to become recordable."""
        var av = TileTensor(self.a, row_major[Self.M_, Self.K_]())
        var bv = TileTensor(self.b, row_major[Self.K_, Self.N_]())
        var cv = TileTensor(self.c, row_major[Self.M_, Self.N_]())
        max_matmul[target="gpu"](cv, av, bv, ctx)

    def wipe(self) raises:
        self.c.enqueue_fill(Scalar[DT](0))
        self.ctx.synchronize()

    def checksum(self, n: Int) raises -> Float64:
        return _sum_first(self.ctx, self.c, Self.M_ * Self.N_, n)

    @staticmethod
    def want(n: Int) -> Float64:
        return Float64(Self.K_) * AVAL * BVAL * Float64(n)


comptime POISON_ELEMS = 512 * 1024 // 4
"""512KB of float32 — the exact `request=512KB` MAX's capture abort named, and
`P * M * N * 4` for arm D's shape at P=2."""


def _pool_recycles(ctx: DeviceContext) raises -> Bool:
    """Does MAX's allocator hand the SAME 512KB block back after a free?

    ⚠⚠ THIS IS ARM D's VACUITY GUARD AND IT IS NOT OPTIONAL. Arm D claims a
    dangling split-K workspace would be caught by allocating over it. That claim
    is worth NOTHING if the pool never reissues the freed block — the churn
    would just grow the heap, the recorded pointer would stay pristine, and arm
    D would PASS by testing nothing. "0 mismatches" == "nothing tested" is the
    default failure mode in this repo, so the churn has to prove it can bite
    before a PASS downstream of it means anything.

    Allocate, record the device pointer, free, allocate the same size again. A
    matching pointer means the pool recycles same-size blocks, which is exactly
    what would land on top of a freed workspace.
    """
    var b1 = ctx.enqueue_create_buffer[DT](POISON_ELEMS)
    ctx.synchronize()
    var p1 = Int(b1.unsafe_ptr())
    _ = b1^
    var b2 = ctx.enqueue_create_buffer[DT](POISON_ELEMS)
    ctx.synchronize()
    var p2 = Int(b2.unsafe_ptr())
    _ = b2^
    return p1 == p2 and p1 != 0


comptime POISON = -12345.0
comptime PROBE_BUFS = 16
comptime PROBE_STRIDE = 64


def _poison_survived(
    ctx: DeviceContext,
    live: List[DeviceBuffer[DT]],
    elems: Int,
    expect_partial: Float64,
) raises -> Int:
    """How many of `live` were WRITTEN THROUGH while the graph replayed.

    Stride-64 scan, not every element: a split-K workspace write is DENSE
    (`P * M * N` contiguous partial sums), so a scan that samples every 64th
    float cannot miss it, and 2048 samples per buffer keeps the check cheap.

    ⚠ AND IT PRINTS THE VALUE IT FOUND, because a count is not an identity.
    "something wrote here" has several possible authors; "this block now holds
    K/P * a * b" has exactly one. `expect_partial` is what a split-K PARTIAL
    SUM must equal — each partition sums K/P terms, so at the arm's constants
    that is 1296 * 0.01 * 0.02 = 0.2592, distinct from both the poison and the
    finished product (0.5184). Matching it identifies the writer as the
    reduction workspace rather than leaving it inferred.
    """
    var host = ctx.enqueue_create_host_buffer[DT](elems)
    var touched = 0
    var reported = False
    for i in range(len(live)):
        ctx.enqueue_copy(host, live[i])
        ctx.synchronize()
        for j in range(0, elems, PROBE_STRIDE):
            if host[j] != Scalar[DT](POISON):
                touched += 1
                if not reported:
                    reported = True
                    print(
                        "      first write: live["
                        + String(i)
                        + "] elem "
                        + String(j)
                        + " = "
                        + String(Float64(host[j]))
                        + "   expected partial-sum signature "
                        + (
                            String(expect_partial)
                            if expect_partial != 0.0
                            else String("n/a (vendor scratch, contents opaque)")
                        )
                        + "   (poison was "
                        + String(POISON)
                        + ")"
                    )
                break
    return touched


def _churn(ctx: DeviceContext) raises:
    """Make the allocator hand out — and dirty — whatever a freed split-K
    workspace left behind.

    ⚠ THIS IS THE WHOLE POINT OF ARM D. A dangling pointer into a pool block
    nobody has touched yet still reads the right bytes, so the first replay can
    pass on a graph that is already wrong. Several ROUNDS of allocate-poison-
    free at exactly the workspace size, so a block that is only reissued after
    a few requests still gets poisoned. `_pool_recycles` is what says this can
    bite at all.
    """
    for _ in range(4):
        var junk = List[DeviceBuffer[DT]]()
        for _ in range(16):
            var b = ctx.enqueue_create_buffer[DT](POISON_ELEMS)
            b.enqueue_fill(Scalar[DT](POISON))
            junk.append(b^)
        ctx.synchronize()
        _ = junk^


def _gemm_arm[
    M_: Int, K_: Int, N_: Int
](
    ctx: DeviceContext,
    label: String,
    churn: Bool,
    parts: Int = 1,
    probe_elems: Int = POISON_ELEMS,
) raises -> Bool:
    var g = _Gemm[M_, K_, N_](ctx)

    # Warm the library OUTSIDE the graph: first-call autotuning, module load
    # and any one-off allocation happen here, not at record time. Then WIPE, so
    # the warm-up's correct output cannot stand in for a graph that recorded
    # nothing.
    g.gemm(ctx)
    ctx.synchronize()
    g.wipe()

    def build(mut builder: DeviceGraphBuilder) raises {mut g}:
        with builder.recording_context() as rec:
            g.gemm(rec)

    var graph = DeviceGraph.create(ctx, build)
    ctx.synchronize()

    var after_build = g.checksum(8)
    var want = _Gemm[M_, K_, N_].want(8)

    graph.replay()
    ctx.synchronize()
    var r1 = g.checksum(8)

    var r2 = Float64(0.0)
    var r3 = Float64(0.0)
    var touched = 0
    if churn:
        # Stage 2 — the freed block, poisoned and handed back.
        g.wipe()
        _churn(ctx)
        graph.replay()
        ctx.synchronize()
        r2 = g.checksum(8)

        # ⚠⚠ STAGE 3 IS THE ONE THAT ACTUALLY BITES, AND STAGE 2 ALONE IS
        # NEARLY VACUOUS WITHOUT IT.
        #
        # `_churn` FREES its buffers before the replay. So at replay time
        # nothing owns that memory, and the split-K kernel WRITES the workspace
        # before `split_k_reduce` READS it — both inside one graph, in recorded
        # order. A dangling-but-still-mapped pointer therefore produces the
        # RIGHT ANSWER: the kernel simply overwrites the poison itself. Stage 2
        # can only catch the block being unmapped, which MAX's pool never does.
        #
        # The failure that matters is the workspace pointer aliasing memory the
        # allocator has since given to someone ELSE and who still holds it. So:
        # allocate poison buffers, KEEP THEM ALIVE across the replay, and check
        # afterwards that nothing wrote through them. A touched buffer means
        # the recorded graph is scribbling on a live allocation and is only
        # "correct" by luck — while corrupting a neighbour.
        g.wipe()
        var live = List[DeviceBuffer[DT]]()
        for _ in range(PROBE_BUFS):
            var b = ctx.enqueue_create_buffer[DT](probe_elems)
            b.enqueue_fill(Scalar[DT](POISON))
            live.append(b^)
        ctx.synchronize()
        graph.replay()
        ctx.synchronize()
        r3 = g.checksum(8)
        # Each of the P partitions sums K/P terms of a*b.
        var expect_partial = (
            Float64(K_) / Float64(parts) * AVAL * BVAL if parts > 1 else 0.0
        )
        touched = _poison_survived(ctx, live, probe_elems, expect_partial)
        _ = live^

    var ok = _close(after_build, 0.0) and _close(r1, want)
    if churn:
        ok = ok and _close(r2, want) and _close(r3, want) and touched == 0

    var detail = (
        "build->"
        + String(after_build)
        + " (want 0)  replay1->"
        + String(r1)
        + "  want "
        + String(want)
    )
    if churn:
        detail += (
            "  replay2(freed churn)->"
            + String(r2)
            + "  replay3(LIVE poison)->"
            + String(r3)
            + "  live buffers written through: "
            + String(touched)
            + "/"
            + String(PROBE_BUFS)
        )
    _verdict(label, ok, detail)

    if churn and touched > 0:
        print("      ^ THE REPLAY WROTE THROUGH A LIVE ALLOCATION. The recorded")
        print("        workspace pointer aliases memory MAX handed to someone")
        print("        else — the GEMM answer is right only by luck, and it is")
        print("        corrupting its neighbour. This is the real dangling")
        print("        workspace, and stage 2 alone cannot see it.")
    elif churn and _close(r1, want) and not _close(r2, want):
        print("      ^ RIGHT then WRONG across an allocator churn: the recorded")
        print("        node holds a pointer MAX already freed AND reused.")
    return ok


def arm_c(ctx: DeviceContext) raises -> Bool:
    print("  C. max_matmul through recording_context — plain multistage")
    return _gemm_arm[OK_M, OK_K, OK_N](
        ctx, "library GEMM, no split-K", churn=False
    )


def arm_d(ctx: DeviceContext) raises -> Bool:
    print("  D. max_matmul through recording_context — SPLIT-K")
    var picked = select_config[DT, DT, DT, False](SK_M, SK_N, SK_K, ctx)
    var parts = Int(picked.num_k_partitions)
    print(
        "     select_config: num_k_partitions = "
        + String(parts)
        + "   block_tile "
        + String(picked.block_tile_shape)
    )
    if parts <= 1:
        # ⚠ A SHAPE IS NOT AN IDENTITY. `select_config` hardcodes A100's
        # sm_count in its wave estimate, so this shape partitioning K is a
        # property of the 5090 it was chosen on, not of the numbers.
        print("     ⚠ K IS NOT PARTITIONED ON THIS DEVICE — arm D would be a")
        print("       second copy of arm C and proves nothing about split-K.")
        print("       Find a shape where this prints > 1 before believing a")
        print("       PASS here. SKIPPED.")
        return True
    var recycles = _pool_recycles(ctx)
    print(
        "     pool reissues a freed "
        + String(POISON_ELEMS * 4 // 1024)
        + "KB block: "
        + ("YES — the churn can bite" if recycles else "NO")
    )
    if not recycles:
        # ⚠ Without this the arm is a vacuity trap: the churn grows the heap,
        # never lands on the freed workspace, and a dangling pointer sails
        # through both replays. A PASS below would then mean NOTHING.
        print("     ⚠ THE CHURN CANNOT REACH A FREED WORKSPACE, so a PASS")
        print("       below is VACUOUS — it would not detect the dangling")
        print("       pointer it exists to detect. Treat D as NO RESULT.")
    var ok = _gemm_arm[SK_M, SK_K, SK_N](
        ctx, "library GEMM, split-K", churn=True, parts=parts
    )
    if ok:
        # A right answer does not prove the recorded call took the split-K
        # branch — `select_config` is a CHOOSER, not the dispatch gate, and the
        # value is identical either way.
        print("     ⚠ CONFIRM THE RECORDED CALL ACTUALLY SPLIT K before")
        print("       believing this. `select_config` is a chooser, not the")
        print("       dispatch gate, and both paths give the same number:")
        print("         pixi run -e nvidia mojo run -I . -D LOGGING_LEVEL=INFO \\")
        print("             benchmarks/bench_device_graph_spike.mojo 2>&1 \\")
        print("           | grep -i 'split_k\\|K partitions'")
    return ok and recycles


# ─────────────────────────────────────────────────────────────────────────────
# Arm E — what it buys
# ─────────────────────────────────────────────────────────────────────────────


def arm_f(ctx: DeviceContext) raises -> Bool:
    """The VENDOR path — the shape ACT's conv GEMMs actually land on.

    ⚠ THIS ARM EXISTS BECAUSE ARMS C AND D CANNOT ANSWER THE QUESTION. Both use
    shapes that PASS `multi_gemm_cond` (`m>1 and n%128==0 and k%32==0 and
    k>=128`, matmul/gpu/__init__.mojo:591), so neither ever reaches the vendor
    branch. `[256 x 64] @ [64 x 256]` fails on `k >= 128` — k=64, exactly ACT's
    `k = OC = 64` — and routes to `matmul_vendor`, which allocates **32MB per
    call** at `blas.mojo:780`.

    Under STREAM capture that allocation raises (empty `graphFreeList`, no
    driver fallback), MAX CATCHES it in a try/except at
    matmul/gpu/__init__.mojo:1373, logs "Vendor BLAS failed", and falls through
    to `matmul_kernel_naive` at BLOCK_DIM=16. Correct results, much slower —
    which is why a stream-captured ACT step comes out SLOWER than eager.

    ⚠ **THE PREDICTION HERE IS THE OPPOSITE, AND IT IS NOT A GUESS ABOUT THE
    POOL.** `recording_context()` does not seed `graphFreeList`; its docstring
    says buffer allocation FORWARDS TO THE BACKING CONTEXT. So the 32MB
    allocation should SUCCEED — as a plain eager allocation, not a graph-scoped
    one — meaning no raise, no "Vendor BLAS failed", and NO naive fallback. The
    32MB is then freed when the call returns while the recorded node keeps its
    pointer, i.e. the same silent aliasing arm D caught, 64x larger.

    So "does DeviceGraph make the allocation problem dissolve?" has a third
    answer besides yes and no: **the allocation stops failing and starts
    corrupting.** The probe therefore runs at 32MB — same-size bucket reuse is
    what `_pool_recycles` demonstrated, so a poison buffer must MATCH the
    allocation to be a plausible recipient of its block.

    Read it with `-D LOGGING_LEVEL=INFO`:
      "Vendor BLAS failed" present  -> the allocation DID fail here too; expect
                                      `matmul_kernel_naive` in the profile and
                                      DeviceGraph inherits the slowdown.
      absent, and `live buffers written through` > 0
                                    -> allocation succeeded and dangles. Faster
                                      than stream capture, and silently wrong.

    ## MEASURED, RTX 5090: NEITHER — it faults outright

        RAISED F: bench_device_graph_spike.mojo:696  CUDA_ERROR_ILLEGAL_ADDRESS

    Line 696 is the `ctx.synchronize()` after the **FIRST** `graph.replay()` —
    before any churn, before a single poison buffer exists. So the recorded
    node dereferences the 32MB scratch and it is already gone. Not the silent
    aliasing of arm D (512KB, which stays mapped in MAX's pool and merely gets
    reissued) and not stream capture's naive fallback: a hard fault on replay 1.

    The fault is STICKY. Everything after it on the same context reports the
    same ILLEGAL_ADDRESS and the allocator starts refusing 4KB requests, which
    is why arm E now runs BEFORE this one.

    ⚠ TO ATTRIBUTE IT, RUN F WITHOUT D:

        NOEIRA_SPIKE_SPLITK=0 pixi run -e nvidia mojo run -I . \\
            benchmarks/bench_device_graph_spike.mojo

    D completes and reports before F starts, so D is probably not the author —
    but "probably" is not attribution, and one env var settles it.
    """
    print("  F. VENDOR path (fails multi_gemm_cond, k=64) — 32MB per call")
    return _gemm_arm[VD_M, VD_K, VD_N](
        ctx,
        "vendor GEMM, 32MB scratch",
        churn=True,
        parts=1,
        probe_elems=VD_PROBE_ELEMS,
    )


def arm_e(ctx: DeviceContext) raises:
    """Eager enqueue of `CHAIN` launches vs one `graph.replay()`, both synced.

    ⚠ THIS IS A LAUNCH-OVERHEAD MEASUREMENT AND NOTHING ELSE. The kernels are
    deliberately trivial, so the ratio here is an upper bound that a real step
    will not see — [[_a_per_call_sweep_is_an_upper_bound_on_a_step]]. Quote it
    as a ceiling.
    """
    print("  E. timing — " + String(CHAIN) + " launches, eager vs replay")

    var buf = ctx.enqueue_create_buffer[DT](N)
    buf.enqueue_fill(Scalar[DT](0))
    ctx.synchronize()
    var lt = tile[N](buf)
    var bump = ctx.compile_function[_bump_lt]()

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        with builder.recording_context() as rec:
            for _ in range(CHAIN):
                rec.enqueue_function(
                    bump, lt, Int32(1), grid_dim=GRID, block_dim=BLOCK
                )

    var graph = DeviceGraph.create(ctx, build)
    ctx.synchronize()

    # Warm both paths before either is timed.
    for _ in range(WARM):
        for _ in range(CHAIN):
            ctx.enqueue_function(
                bump, lt, Int32(1), grid_dim=GRID, block_dim=BLOCK
            )
        ctx.synchronize()
        graph.replay()
        ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(TIMED_ITERS):
        for _ in range(CHAIN):
            ctx.enqueue_function(
                bump, lt, Int32(1), grid_dim=GRID, block_dim=BLOCK
            )
        ctx.synchronize()
    var t1 = perf_counter_ns()
    for _ in range(TIMED_ITERS):
        graph.replay()
        ctx.synchronize()
    var t2 = perf_counter_ns()

    var eager_us = Float64(t1 - t0) / 1000.0 / Float64(TIMED_ITERS)
    var replay_us = Float64(t2 - t1) / 1000.0 / Float64(TIMED_ITERS)
    # ⚠ ROUNDED ON PURPOSE. `String(Float64)` prints ~16 significant digits, and
    # on the 5090 the ratio's tail ran over the following line in the captured
    # output ("3.6926628022169els — a CEILING..."). Two decimals is more than
    # this measurement supports anyway.
    print(
        "     eager  "
        + _fixed2(eager_us)
        + " us/iter    replay "
        + _fixed2(replay_us)
        + " us/iter    "
        + _fixed2(eager_us / replay_us)
        + "x"
    )
    print(
        "     per launch: "
        + _fixed2(eager_us / Float64(CHAIN))
        + " -> "
        + _fixed2(replay_us / Float64(CHAIN))
        + " us  (replay also carries ONE sync for the whole chain)"
    )
    print("     ⚠ trivial kernels: a CEILING on launch overhead, not a step.")

    # ⚠⚠ THIS READ-BACK IS LOAD-BEARING TWICE OVER, AND ITS ABSENCE CRASHED THE
    # FIRST 5090 RUN with CUDA_ERROR_ILLEGAL_ADDRESS surfacing at
    # `AsyncRT_DeviceContext_release`.
    #
    # (a) LIFETIME. Mojo destroys a value at its LAST USE. Without a mention of
    #     `buf` down here, its last use is the `LayoutTensor` construction
    #     above — so the DeviceBuffer is freed there and every launch in this
    #     arm, eager and replayed alike, writes through a dangling `lt`. Arms A
    #     and B never hit it only because they happen to read `buf` back at the
    #     end. `noeira/cuda/graph.mojo:54` and this file's own `_Gemm` docstring
    #     both state the rule; the timing arm was written without applying it.
    #
    # (b) VACUITY. A timing arm that never checks a value is happy to report a
    #     beautiful speedup for launches that did nothing. WARM eager + WARM
    #     replayed + TIMED eager + TIMED replayed passes of CHAIN increments is
    #     an exact integer, so it either lands on it or something did not run.
    comptime want = Float64(2 * (WARM + TIMED_ITERS) * CHAIN)
    var got = _read_first(ctx, buf)
    _verdict(
        "launches actually ran",
        _close(got, want),
        "counter->" + String(got) + "  want " + String(want),
    )


# ─────────────────────────────────────────────────────────────────────────────


def _run[
    arm: def (DeviceContext) raises thin -> Bool
](name: String, ctx: DeviceContext) -> Int:
    """Run one arm. Returns 1 pass, 0 fail, -1 raised.

    ⚠ A RAISE IS ITS OWN OUTCOME, not a failure. Off NVIDIA every arm is
    EXPECTED to raise (Metal has neither stream capture nor device graphs), and
    on NVIDIA the exception TEXT is the finding — `recording_context()` raises
    by design on host-visible waits, so a library path that synchronizes shows
    up here and nowhere else.
    """
    try:
        return 1 if arm(ctx) else 0
    except e:
        print("    RAISED  " + name + ": " + String(e))
        return -1


def main() raises:
    var ctx = DeviceContext()
    print("=" * 76)
    print("MAX DeviceGraph spike —", ctx.name())
    print("=" * 76)
    print(
        "  0. import + link:  max.gpu.host.DeviceGraph resolved, context live."
    )
    print("     This file calls NO interceptor and borrows NO stream.")
    print("     ⚠ Under `pixi run -e nvidia` the activation preloads")
    print("       libcuda_intercept.so anyway — a '[intercept] ...' line below")
    print("       is THAT, not this file. Build and run the binary directly")
    print("       (see the header) to actually demonstrate independence.")
    comptime if not default_accelerator().is_nvidia_gpu():
        print("")
        print("  ⚠ NOT AN NVIDIA DEVICE. Metal supports neither stream capture")
        print("    nor device graphs, so every arm below is expected to RAISE")
        print("    and NO VERDICT IS POSSIBLE. This run only proves the file")
        print("    builds and links. Run it on the 5090 for the answer.")
    print("")

    var a = _run[arm_a]("A", ctx)
    var b = _run[arm_b]("B", ctx)
    var c = _run[arm_c]("C", ctx)

    # ⚠ E RUNS BEFORE D AND F ON PURPOSE. A CUDA error is STICKY: once D or F
    # faults, every later call on this context reports the same
    # ILLEGAL_ADDRESS, the allocator starts refusing 4KB requests, and E's
    # "failure" is collateral rather than a result. It cost a whole arm once.
    print("")
    try:
        arm_e(ctx)
    except e:
        print("  E. RAISED: " + String(e))
    print("")

    var d = 2  # 2 = not attempted
    if getenv("NOEIRA_SPIKE_SPLITK", "1") == "0":
        print("  D. SKIPPED by NOEIRA_SPIKE_SPLITK=0")
    else:
        print("  (D and F may fault or abort — everything above has printed.")
        print("   NOEIRA_SPIKE_SPLITK=0 / NOEIRA_SPIKE_VENDOR=0 skip them.)")
        d = _run[arm_d]("D", ctx)

    var f = 2
    if getenv("NOEIRA_SPIKE_VENDOR", "1") == "0":
        print("  F. SKIPPED by NOEIRA_SPIKE_VENDOR=0")
    else:
        f = _run[arm_f]("F", ctx)

    print("")
    print("=" * 76)
    comptime if not default_accelerator().is_nvidia_gpu():
        print("  NO VERDICT — not an NVIDIA device.")
        return
    if a != 1:
        print("  A FAILED -> DeviceGraph is not usable here at all; B/C/D say")
        print("  nothing. Check the MAX version exports it:")
        print("    grep -n DeviceGraph .pixi/envs/nvidia/lib/mojo/max.mojoc")
    elif b != 1:
        print("  A passes, B fails -> the explicit DAG works but")
        print("  recording_context does not, so a STEP() closure cannot be")
        print("  recorded as-is and every node would have to be hand-added.")
        print("  That is a much larger migration than the shim is worth.")
    elif c != 1:
        print("  A,B pass, C fails -> hand kernels record, MAX library calls")
        print("  do not. Our steps are mostly library GEMMs, so read the RAISED")
        print("  text above: that names the call DeviceGraph will not record.")
    else:
        print("  A,B,C PASS -> DeviceGraph records our kernels AND MAX's")
        print("  library calls, with no interceptor. `noeira/cuda/graph.mojo`")
        print("  is replaceable; the migration is giving `maybe_capture_replay`")
        print("  a STEP(ctx) signature instead of a captured context.")
        if f == -1:
            print("")
            print("  F RAISED -> the vendor path's 32MB scratch is GONE by")
            print("  replay time: an illegal access on the FIRST replay, before")
            print("  any churn. Not arm D's silent aliasing and not stream")
            print("  capture's naive fallback — a hard fault. And that is the")
            print("  shape ACT's conv GEMMs land on (k = OC = 64 fails")
            print("  multi_gemm_cond), so padding K past 128 to dodge the")
            print("  vendor path is the prerequisite, the same way owning the")
            print("  workspace is for split-K.")
            print("  Confirm it is F's own and not D's residue:")
            print("    NOEIRA_SPIKE_SPLITK=0 <same command>")
        if f == 0:
            print("")
            print("  F FAILED -> the VENDOR path's 32MB scratch dangles too,")
            print("  and that is the shape ACT's conv GEMMs actually land on")
            print("  (k = OC = 64 fails multi_gemm_cond). Note this is NOT the")
            print("  stream-capture failure: there the allocation RAISES, MAX")
            print("  catches it and drops to matmul_kernel_naive — correct but")
            print("  slow. Here it SUCCEEDS eagerly and dangles. Check the")
            print("  INFO log for 'Vendor BLAS failed' to tell them apart.")
        elif f == 1:
            print("")
            print("  F PASSED -> the vendor path records cleanly. Check the")
            print("  INFO log for 'Vendor BLAS failed' anyway: a PASS with that")
            print("  line present means MAX fell back to the naive kernel and")
            print("  the arm is measuring the SLOW path being correct.")
        if d == 0:
            print("")
            print("  D FAILED -> split-K is STILL not recordable, and here it")
            print("  fails WORSE than under stream capture. Stream capture")
            print("  ABORTS on this shape; DeviceGraph returns the RIGHT")
            print("  ANSWER and silently writes over whatever the allocator")
            print("  handed the freed workspace to next. Nothing raises.")
            print("")
            print("  ACT issues ~46 split-K GEMMs per step, so recording one")
            print("  would plant 46 stray writers in live memory and the loss")
            print("  would simply be wrong. Own the workspace first — that")
            print("  work is already landed for Linear/Conv2D dW — and only")
            print("  then record a step that contains one.")
        elif d == 1:
            print("")
            print("  D PASSED, and passed with the churn PROVEN able to bite")
            print("  (`pool reissues a freed 512KB block: YES` above), so this")
            print("  is not the vacuous pass. The split-K workspace survives")
            print("  recording, which stream capture cannot even attempt —")
            print("  it aborts the process on this exact shape.")
            print("")
            print("  and `live buffers written through: 0/16` says the")
            print("  recorded workspace does not alias a live allocation —")
            print("  which is the only stage that can actually catch a")
            print("  dangling workspace, since stages 1-2 free their poison")
            print("  before replaying and the kernel just overwrites it.")
            print("")
            print("  Still worth confirming ONCE that the RECORDED call splits")
            print("  K at all — select_config is a chooser, not the dispatch")
            print("  gate, and both branches print the same number:")
            print("    pixi run -e nvidia mojo run -I . -D LOGGING_LEVEL=INFO \\")
            print("        benchmarks/bench_device_graph_spike.mojo 2>&1 \\")
            print("      | grep -i 'split_k\\|K partitions'")
            print("  Expect 1,1,2,2 — arm C warm+recorded, arm D warm+recorded.")
