"""Does `linalg.matmul` care whether the CONTRACTION dim K is a multiple of 16?

    pixi run -e apple  mojo run -I . benchmarks/bench_matmul_k_alignment.mojo
    pixi run -e nvidia mojo run -I . benchmarks/bench_matmul_k_alignment.mojo

## Why this exists

On Apple/Metal (M1 Pro, 2026-08-12) `max_matmul` falls off a ~10x cliff the
moment `K % 16 != 0`, and it is the dominant cost of TD-MPC2's MPPI planner and
of its world-model training rollout:

    [268 x 512] @ [512 x 512]   162 us    867 GFLOPS
    [268 x 518] @ [518 x 512]  1339 us    106 GFLOPS     <- same FLOPs, 8x slower
    [268 x 528] @ [528 x 512]   143 us   1011 GFLOPS

518 is not an arbitrary number: it is `ZA = LATENT + ACT` = 512 + 6, the input
width of TD-MPC2's dynamics trunk, reward trunk and every Q head. Any
`(obs | act)` concatenation feeding a critic has the same problem, and so does
`BINS = 101` whenever a two-hot logit vector is used as an input width.

## The decision this benchmark exists to make

Padding K costs something when it does not help: a copy kernel and a padded
weight slab per `Linear` whose `IN_ % 16 != 0`. So the padding policy in
`noeira/nn/primitives/linear.mojo` should be keyed on a MEASUREMENT, not on a
guess, and not on the platform. **The number that decides it is the RATIO**
(unaligned time / padded time) — it is backend-independent, unlike GFLOPS,
which needs a per-device peak to interpret.

    ratio >> 1  ->  the cliff is real here; pad.
    ratio ~ 1   ->  the library already absorbs unaligned K; do not pad,
                    the copy would be pure overhead.
    ratio < 1   ->  padding COSTS time here; definitely do not pad.

⚠ The Apple numbers quoted above are a REFERENCE, not an expectation. cuBLAS
pads internally and may well absorb this entirely, in which case the right
answer is an Apple-only default — but that is exactly what has not been
measured yet, and a permanent platform branch is too expensive to add on a
hunch.

⚠ This measures `linalg.matmul` DIRECTLY on raw device buffers, so nothing here
is attributable to the `Tensor` / `Param` plumbing in `nn`. The isolated
per-GEMM saving also does NOT transfer 1:1 to an end-to-end agent number — see
the assessment in the commit that added this file: padding bought 2.27x on the
full TD-MPC2 planner at 1024 samples but only 1.21x at 256, where the
~700-launch dependent kernel chain becomes the binding constraint instead.
"""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from linalg.matmul import matmul as max_matmul

comptime DT = DType.float32


def time_shape[M: Int, K: Int, N: Int](
    ctx: DeviceContext, reps: Int
) raises -> Float64:
    """Microseconds per `max_matmul` at [M x K] @ [K x N], warmed up.

    One `synchronize()` for the whole batch of `reps`, not one per call: a
    per-call sync measures the ~190 us round-trip latency instead of the work.
    """
    var a = ctx.enqueue_create_buffer[DT](M * K)
    var b = ctx.enqueue_create_buffer[DT](K * N)
    var c = ctx.enqueue_create_buffer[DT](M * N)
    a.enqueue_fill(Scalar[DT](0.01))
    b.enqueue_fill(Scalar[DT](0.01))
    ctx.synchronize()

    var av = TileTensor(a, row_major[M, K]())
    var bv = TileTensor(b, row_major[K, N]())
    var cv = TileTensor(c, row_major[M, N]())

    # Warm up: the first call compiles/selects the pipeline state.
    for _ in range(10):
        max_matmul[target="gpu"](cv, av, bv, ctx)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(reps):
        max_matmul[target="gpu"](cv, av, bv, ctx)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    return Float64(t1 - t0) / 1000.0 / Float64(reps)


def report[M: Int, K: Int, N: Int](ctx: DeviceContext, reps: Int) raises:
    var us = time_shape[M, K, N](ctx, reps)
    var gflops = 2.0 * Float64(M) * Float64(K) * Float64(N) / us / 1e3
    print(
        "   [", M, " x ", K, "] @ [", K, " x ", N, "]   K%16=", K % 16,
        "   ", us, " us   ", gflops, " GFLOPS", sep="",
    )


def pair[M: Int, K: Int, K16: Int, K32: Int, K64: Int, N: Int](
    ctx: DeviceContext, note: String, reps: Int
) raises:
    """The decision line: one real shape against SEVERAL padding targets.

    ⚠ The first version of this benchmark tested round-up-to-16 ONLY, and that
    made the RTX 5090 look like it had no cliff at all: K=518 is 8x slower than
    K=512 there, but K=528 (round-16) is *also* on the slow path, so the ratio
    came out ~1.0 and read as "nothing to fix". K=544 (round-32) is fast. A
    single hardcoded target cannot answer "should we pad" — it can only answer
    "is THIS target good", which is a different and much less useful question.

    Every target is >= K, so every padded call does strictly MORE arithmetic;
    any ratio above 1 is a pure win and understates the real benefit.
    """
    var raw = time_shape[M, K, N](ctx, reps)
    var p16 = time_shape[M, K16, N](ctx, reps)
    var p32 = time_shape[M, K32, N](ctx, reps)
    var p64 = time_shape[M, K64, N](ctx, reps)

    var best = p16
    var best_k = K16
    if p32 < best:
        best = p32
        best_k = K32
    if p64 < best:
        best = p64
        best_k = K64
    var ratio = raw / best
    # 1.2x is inside the run-to-run spread seen on the 5090 — do not call
    # anything below it a win.
    var verdict = String("PAD") if ratio > 1.25 else (
        String("no-op") if ratio > 0.85 else String("PADDING COSTS")
    )
    print("   ", note, sep="")
    print(
        "      K=", K, " (%16=", K % 16, " %32=", K % 32, "): ", raw,
        " us   |  ->", K16, ": ", p16, "   ->", K32, ": ", p32,
        "   ->", K64, ": ", p64, sep="",
    )
    print(
        "      best target K=", best_k, "  ratio=", ratio, "x   ", verdict,
        sep="",
    )


def main() raises:
    var ctx = DeviceContext()
    print("=" * 78)
    print("max_matmul K-alignment — device:", ctx.name())
    print("=" * 78)
    print()

    print("A. K sweep at the TD-MPC2 planner's shape (M=268, N=512)")
    print("   Apple  : 512/528/544 ~1000 GFLOPS; 513..524 ~100.  (rule: K%16)")
    print("   RTX5090: 512/544 ~5500 GFLOPS; 513..528 ~680.      (rule: K%32)")
    print("   ⚠ 528 is %16=0 but %32=16 — fast on Metal, SLOW on the 5090.")
    report[268, 512, 512](ctx, 200)
    report[268, 513, 512](ctx, 200)
    report[268, 514, 512](ctx, 200)
    report[268, 516, 512](ctx, 200)
    report[268, 518, 512](ctx, 200)   # <- TD-MPC2 walker: ZA = 512 + 6
    report[268, 520, 512](ctx, 200)   # <- multiple of 8, still slow on Metal
    report[268, 524, 512](ctx, 200)
    report[268, 528, 512](ctx, 200)   # <- %16=0, %32=16: THE DISCRIMINATOR
    report[268, 544, 512](ctx, 200)
    print()

    print("A2. Is the rule K%32? Multiples of 32 vs the %16-only values")
    print("    If %32 is the rule, the left column is fast and the right slow.")
    report[268, 576, 512](ctx, 200)   # %32=0
    report[268, 560, 512](ctx, 200)   # %16=0, %32=16
    report[268, 640, 512](ctx, 200)   # %32=0
    report[268, 624, 512](ctx, 200)   # %16=0, %32=16
    print()

    print("B. Controls — is it K only? Sweep M and N, hold K aligned.")
    print("   Apple  : neither M nor N matters; only K.")
    print("   RTX5090: N MATTERS TOO (96/101/518 slow, 128/512 fast).")
    report[256, 512, 512](ctx, 200)
    report[264, 512, 512](ctx, 200)
    report[272, 512, 512](ctx, 200)
    report[268, 512, 96](ctx, 200)
    report[268, 512, 101](ctx, 200)
    report[268, 512, 128](ctx, 200)
    report[256, 512, 518](ctx, 200)   # unaligned N, aligned K
    print()

    print("C. THE DECISION — real repo shapes vs SEVERAL padding targets")
    print("   Each shape is padded up to the next multiple of 16, 32 and 64;")
    print("   the best of the three sets the ratio. A single target cannot")
    print("   answer this — see the note on `pair`.")
    pair[268, 518, 528, 544, 576, 512](
        ctx, "TD-MPC2 MPPI plan (za = latent|act)", 200
    )
    pair[256, 518, 528, 544, 576, 512](
        ctx, "TD-MPC2 TRAINING batch (same za)", 200
    )
    pair[256, 264, 272, 288, 320, 512](ctx, "larger latent|act concat", 200)
    pair[256, 101, 112, 128, 128, 512](
        ctx, "BINS=101 used as an input width", 300
    )
    pair[256, 30, 32, 32, 64, 256](
        ctx, "SAC/TD3 critic, walker obs|act = 30", 300
    )
    pair[256, 24, 32, 32, 64, 256](ctx, "walker obs alone = 24", 300)
    print()

    print("=" * 78)
    print("Read the `best target` lines in C. If the winning target is the")
    print("same on both backends, the padding policy in linear.mojo needs no")
    print("platform gate — just round K up to that multiple when IN_ is off it.")
    print("If they disagree, the TARGET is platform-specific, not the decision")
    print("to pad at all.")
    print("=" * 78)
