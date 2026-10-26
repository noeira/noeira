"""Sweep the real ACT dW shapes: which split K, and what does owning the
workspace buy on each?

`bench_splitk_persistent_workspace.mojo` proved the technique on ONE shape
(1.39x, bit-identical, captures and replays). This asks whether the win holds
across the shapes ACT actually issues, and turns "1.39x on a GEMM" into "N ms
per training step".

WHERE THE SHAPES COME FROM
--------------------------
`Linear`'s GPU vjp issues its weight gradient as

    max_matmul(dW[K_PAD, N_PAD], cacheT[K_PAD, B], go[B, N_PAD])   (linear.mojo:922)

so a dW GEMM is  M = in_features, N = out_features, K = B, where B is the
FLATTENED row count = batch x tokens. Everything below follows from
`examples/so101/act_so101_profile_gpu.mojo` (BATCH=16, DIM=256, FF=1024, K=60,
N_ENC=4, N_DEC=1, 2 cameras at 240x320) and `deep_agents/act/layers.mojo`:

    transformer encoder   162 tokens (2 x 8x10 ResNet18 features + latent + qpos)
                          B = 16 * 162 = 2592
    CVAE encoder           62 tokens (cls + qpos + 60 actions)
                          B = 16 *  62 =  992
    decoder self/cross-q   60 queries          B = 16 *  60 =  960
    decoder cross k/v     162 memory tokens    B = 16 * 162 = 2592

⚠ That derivation is not free-floating: B=2592, B=992 and B=960 each reproduce
a shape MODULAR_MATMUL_ALLOC_REPORT.md measured independently
(`[256 x 2592] @ [2592 x 256]`, `[992 x 256] @ [256 x 256]`,
`[960 x 256] @ [256 x 32]`). Rows marked MEASURED appear in that report
verbatim; rows marked DERIVED follow from the config and the layer list.

⚠ The AUTHORITATIVE list is the training step itself. To check this table is
complete rather than merely correct:

    pixi run -e nvidia mojo run -D LOGGING_LEVEL=INFO -I . \
        examples/so101/act_so101_profile_gpu.mojo 2>&1 \
      | grep -A6 'MATMUL GPU execution started' \
      | grep -E 'MxNxK|K partitions' | paste - - | sort | uniq -c | sort -rn

`q`, `k` and `v` are SEPARATE Linears here, not a fused `Linear[DIM, 3*DIM]`
(layers.mojo:15 says so explicitly), which is why the count column below has
four 256x256 attention GEMMs per layer rather than one wide one.

Run (NVIDIA only):

    pixi run -e nvidia mojo run -I . benchmarks/bench_splitk_act_dw_sweep.mojo
"""

from std.math import ceildiv, align_up
from std.sys import default_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer, FuncAttribute
from layout import Coord, Idx, TileTensor, row_major
from std.utils.index import Index

from linalg.matmul import matmul as max_matmul
from linalg.utils_gpu import MatmulConfig, MatmulKernels, select_config
from linalg.matmul.gpu import multistage_gemm_split_k_kernel, split_k_reduce


comptime DT = DType.float32
comptime WARMUP = 5
comptime REPS = 30

comptime PS = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 23, 24, 33, 40, 41]
"""Candidate partition counts. Deliberately includes illegal ones (16 and 20
overrun at K=2592; 23 and 40 undercover) so the sweep prints WHICH rule each
violates instead of quietly omitting it."""


struct SplitKWorkspace[dtype: DType](Movable):
    """A device buffer reused across split-K GEMMs. See
    `bench_splitk_persistent_workspace.mojo` for the why."""

    var buf: DeviceBuffer[Self.dtype]
    var capacity: Int

    def __init__(out self, ctx: DeviceContext, capacity: Int) raises:
        self.buf = ctx.enqueue_create_buffer[Self.dtype](capacity)
        self.capacity = capacity
        ctx.synchronize()


def splitk_gemm[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    ws_type: DType, //,
    *,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[mut=False, a_type, ...],
    b: TileTensor[mut=False, b_type, ...],
    num_partitions: Int,
    mut ws: SplitKWorkspace[ws_type],
    ctx: DeviceContext,
) raises:
    """`multistage_gemm`'s split-K branch on a caller-owned workspace.
    Mirrors matmul/gpu/__init__.mojo:1840-1915 without its per-call
    `enqueue_create_buffer` / `_ = work_space_data^` pair."""
    # The workspace dtype is `config.split_k_reduction_type` -- taken as an
    # INFERRED parameter rather than written into the signature, because the
    # compiler will not fold `config.split_k_reduction_type` when the caller
    # holds the workspace in a variable typed elsewhere. The assert restores
    # exactly the guarantee the signature would have given.
    comptime assert ws_type == config.split_k_reduction_type, (
        "workspace dtype must equal config.split_k_reduction_type"
    )
    var M = Int(c.dim[0]())
    var N = Int(c.dim[1]())

    if num_partitions * M * N > ws.capacity:
        raise Error("SplitKWorkspace too small for this GEMM")

    # ⚠ PARTITION COUNT SAFETY. `multistage_gemm_split_k_kernel`'s NVIDIA path
    # splits K with `LayoutTensor.split[axis, split_alignment=BK]`, which is
    # (layout_tensor.mojo:3870):
    #
    #     part = align_up(K // P, BK)                     <- FLOOR div, then align
    #     size_of_partition_i = min(part, K - i * part)
    #     ptr_i               = base + i * part * stride
    #
    # so when `(P-1) * part >= K` the last block's pointer starts PAST the end
    # of A and B and its size goes NEGATIVE. Measured on the 5090 at
    # K=2592, BK=16: P<=12 is fine (11*224 = 2464 < 2592) and P=16 faults
    # (15*176 = 2640 > 2592, last size -48) with CUDA_ERROR_ILLEGAL_ADDRESS.
    #
    # There are TWO ways to get this wrong, and only one of them is loud:
    #
    #   OVERRUN       (P-1) * part >= K   the last block starts past the end of
    #                                     A and B with a NEGATIVE size
    #                                     -> CUDA_ERROR_ILLEGAL_ADDRESS
    #   UNDERCOVERAGE  P * part < K       `part` is floor-then-align, so the P
    #                                     partitions can fail to span K. The
    #                                     tail is never accumulated and the
    #                                     GEMM SILENTLY RETURNS A WRONG ANSWER
    #
    # Both measured on the 5090 at K=2592, BK=16. P=16 overruns
    # (15*176 = 2640 > 2592). P=40 undercovers: part = align_up(64,16) = 64 and
    # 40*64 = 2560, so 32 of 2592 contraction elements are dropped -- and the
    # observed relative error was 0.012344, against 32/2592 = 0.012346. The
    # error IS the fraction of K dropped, to five digits.
    #
    # ⚠ MAX'S OWN GUARD COVERS NEITHER. `select_config` breaks on `K < P * bk`
    # (2592 < 256 is false at P=16, so it passes) and is saved only by the
    # SEPARATE `min_k_partition = 1024` test capping P at 2 here for unrelated
    # reasons. Any caller that chooses P itself -- us, or MAX's own
    # `TUNE_NUM_K_PARTITIONS` autotune define -- can reach both.
    var K_dim = Int(a.dim[1]())
    comptime BK = config.block_tile_shape[2]
    var part = align_up(K_dim // num_partitions, BK) if num_partitions > 0 else 0
    if num_partitions < 1:
        raise Error("num_k_partitions must be >= 1")
    if (num_partitions - 1) * part >= K_dim:
        raise Error(
            "num_k_partitions OVERRUNS K: the first P-1 partitions of"
            " align_up(K//P, BK) already cover K, so the last block would read"
            " past the operands (this one crashes loudly)"
        )
    if num_partitions * part < K_dim:
        raise Error(
            "num_k_partitions UNDERCOVERS K: P*align_up(K//P, BK) < K, so the"
            " tail of the contraction is never accumulated and the GEMM"
            " silently returns a wrong answer"
        )


    # MAX 26.7: the kernel takes TileTensors, with a LayoutType and a
    # linear_idx_type parameter per operand (same as nn/core/splitk_gemm.mojo).
    comptime static_N = c.static_shape[1]
    var ws_tt = TileTensor(ws.buf, row_major(Coord(num_partitions, M, Idx[static_N])))

    comptime kern = multistage_gemm_split_k_kernel[
        c_type, type_of(c).LayoutType,
        a_type, type_of(a).LayoutType,
        b_type, type_of(b).LayoutType,
        ws_type, type_of(ws_tt).LayoutType,
        transpose_b,
        type_of(c).linear_idx_type,
        type_of(a).linear_idx_type,
        type_of(b).linear_idx_type,
        type_of(ws_tt).linear_idx_type,
        config,
    ]

    ctx.enqueue_function[kern](
        c.as_unsafe_any_origin(),
        a.as_unsafe_any_origin(),
        b.as_unsafe_any_origin(),
        ws_tt.as_unsafe_any_origin(),
        Int32(num_partitions),
        grid_dim=(
            ceildiv(N, config.block_tile_shape[1]),
            ceildiv(M, config.block_tile_shape[0]),
            num_partitions,
        ),
        block_dim=config.block_dim(),
        shared_mem_bytes=config.shared_mem_usage(),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(config.shared_mem_usage())
        ),
    )

    split_k_reduce(c, ws_tt, ctx)


def sweep_one[
    M: Int, K: Int, N: Int, COUNT: Int, LABEL: StaticString
](ctx: DeviceContext, mut ws: SplitKWorkspace[DT], mut total_saved_us: Float64) raises:
    """One row of the sweep. Prints the dispatch verdict, and A/B only when
    the shape actually partitions K -- a shape that does not split takes
    plain `multistage_gemm` today and has nothing to win."""
    var picked = select_config[DT, DT, DT, False](M, N, K, ctx)
    var P = picked.num_k_partitions

    if P <= 1:
        print(
            "  ", LABEL, " [", M, "x", K, "] @ [", K, "x", N, "]",
            "  x", COUNT, "  ->  no split (P=1), unchanged", sep="",
        )
        return

    var ab = ctx.enqueue_create_buffer[DT](M * K)
    var bb = ctx.enqueue_create_buffer[DT](K * N)
    var c_ref = ctx.enqueue_create_buffer[DT](M * N)
    var c_ours = ctx.enqueue_create_buffer[DT](M * N)
    ab.enqueue_fill(Float32(0.01))
    bb.enqueue_fill(Float32(0.02))
    c_ref.enqueue_fill(Float32(0.0))
    c_ours.enqueue_fill(Float32(0.0))
    ctx.synchronize()

    var av = TileTensor(ab, row_major[M, K]())
    var bv = TileTensor(bb, row_major[K, N]())
    var cref = TileTensor(c_ref, row_major[M, N]())
    var cours = TileTensor(c_ours, row_major[M, N]())

    # ---- arm A: linalg.matmul (allocates its workspace per call) ----------
    for _ in range(WARMUP):
        max_matmul[target="gpu"](cref, av, bv, ctx)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(REPS):
        max_matmul[target="gpu"](cref, av, bv, ctx)
    ctx.synchronize()
    var t1 = perf_counter_ns()

    # ---- arm B: ours, on the persistent workspace ------------------------
    # `cfg` supplies the grid and the shared-memory request, so it must be the
    # tile `select_config` chose. MAX picks between three; `_256x128_3` is
    # A100-gated, so two branches cover every NVIDIA part we run on. Getting
    # this wrong is a launch-geometry mismatch, not a slowdown -- hence the
    # raise rather than a fallback.
    comptime kernels = MatmulKernels[DT, DT, DT, False]()
    var t2: Int
    var t3: Int
    if picked == kernels.ampere_256x64_4:
        comptime cfg = kernels.ampere_256x64_4
        for _ in range(WARMUP):
            splitk_gemm[transpose_b=False, config=cfg](cours, av, bv, P, ws, ctx)
        ctx.synchronize()
        t2 = perf_counter_ns()
        for _ in range(REPS):
            splitk_gemm[transpose_b=False, config=cfg](cours, av, bv, P, ws, ctx)
        ctx.synchronize()
        t3 = perf_counter_ns()
    elif picked == kernels.ampere_128x128_4:
        comptime cfg = kernels.ampere_128x128_4
        for _ in range(WARMUP):
            splitk_gemm[transpose_b=False, config=cfg](cours, av, bv, P, ws, ctx)
        ctx.synchronize()
        t2 = perf_counter_ns()
        for _ in range(REPS):
            splitk_gemm[transpose_b=False, config=cfg](cours, av, bv, P, ws, ctx)
        ctx.synchronize()
        t3 = perf_counter_ns()
    else:
        raise Error(
            "select_config picked a tile this sweep does not instantiate"
            " (likely ampere_256x128_3, which is A100-gated)"
        )

    var us_a = Float64(t1 - t0) / 1000.0 / Float64(REPS)
    var us_b = Float64(t3 - t2) / 1000.0 / Float64(REPS)

    var worst = Float64(0.0)
    with c_ref.map_to_host() as hr:
        with c_ours.map_to_host() as ho:
            for i in range(M * N):
                var d = abs(Float64(hr[i]) - Float64(ho[i]))
                if d > worst:
                    worst = d

    var saved = (us_a - us_b) * Float64(COUNT)
    total_saved_us += saved

    print(
        "  ", LABEL, " [", M, "x", K, "] @ [", K, "x", N, "]",
        "  x", COUNT, "  P=", P,
        "  A ", us_a, "us  B ", us_b, "us  ", us_a / us_b, "x",
        "  |A-B| ", worst,
        "  saves ", saved, "us/step",
        sep="",
    )

    # Keep every operand alive past the last launch. A view does not own its
    # buffer, and Mojo destroys at last use -- see bench_splitk_persistent_
    # workspace.mojo's keep-alive note, which cost one CUDA_ERROR_ILLEGAL_ADDRESS
    # to learn.
    _ = ab^
    _ = bb^
    _ = c_ref^
    _ = c_ours^


def choose_partitions(
    M: Int, N: Int, K: Int, BM: Int, BN: Int, BK: Int, sm_count: Int,
    max_p: Int = 48,
) -> Int:
    """Pick `num_k_partitions` to fill the machine once, and no more.

    The sweep on a 5090 (170 SMs) puts the knee exactly at the wave boundary.
    Blocks launched is `tiles * P` where `tiles = ceildiv(M,BM)*ceildiv(N,BN)`:

        [256 x 2592] @ [2592 x  256]   4 tiles   P=24 ->  96 blocks   11.23 us
        [256 x 2592] @ [2592 x 1024]  16 tiles   P= 8 -> 128 blocks   20.79 us
                                                 P=12 -> 192 blocks   28.75 us

    128 blocks fit in one wave on 170 SMs and 192 do not, and the 192-block
    point is 38% SLOWER than the 128-block one despite doing the same work in
    more parallel pieces. So: maximise P subject to `tiles * P <= sm_count`.

    ⚠ LEGALITY IS NOT MONOTONE IN P, so this scans instead of breaking. A P is
    usable only if it BOTH avoids the overrun and covers K (see `splitk_gemm`):

        (P - 1) * part <  K        no block starts past the end
        P       * part >= K        the partitions actually span K

    At K=2592, BK=16 that leaves 1..15, 17, 18, 21, 24, 27, 33, 41. P=16 and 20
    overrun; P=23 and 40 UNDERCOVER, which does not crash -- it silently drops
    the tail of the contraction. A loop that stopped at the first failure would
    return 15 and miss the best point by a wide margin.

    ⚠ MEASURED ACCURACY: right for the 16-tile shapes (picks P=10, which is
    the measured optimum at 19.1 us) and ~7% off for the 4-tile one (picks
    P=41 / 11.87 us where P=33 / 11.06 us wins). It only models the GEMM side.
    The reduce reads `P * M * N`, so its cost grows LINEARLY in P: on the
    4-tile shape that is ~5.8 us of an 11.06 us total at P=33, and past there
    another partition costs more reduce traffic than it saves GEMM time.

    Modelling that second term from three shapes would be curve-fitting. Use
    `autotune_partitions` where a measurement is possible -- the shapes are
    comptime-fixed, so one sweep at init settles it -- and keep this as the
    no-measurement fallback. It is never WRONG, only up to ~7% slow.

    Returns 1 when nothing better is available, which callers should read as
    "do not use split-K for this shape".
    """
    var tiles = ceildiv(M, BM) * ceildiv(N, BN)
    var best = 1
    for P in range(2, max_p + 1):
        if tiles * P > sm_count:
            continue
        var part = align_up(K // P, BK)
        if (P - 1) * part >= K:
            continue                      # overruns: would fault
        if P * part < K:
            continue                      # undercovers: would be WRONG
        if P > best:
            best = P
    return best


def autotune_partitions[
    M: Int, K: Int, N: Int
](ctx: DeviceContext, mut ws: SplitKWorkspace[DT], reps: Int = 20) raises -> Int:
    """Measure every legal P once and return the fastest.

    This is what an integration should call at model init. The dW shapes are
    comptime constants, the candidate set is small, and one sweep costs a few
    milliseconds against a training run of hours -- so there is no reason to
    infer the partition count from a formula when it can be observed.

    Only legal P are tried: `(P-1)*part < K` (no overrun) and `P*part >= K`
    (covers K), both evaluated at comptime, so an illegal candidate is never
    instantiated. See `splitk_gemm` for why both rules exist.
    """
    comptime kernels = MatmulKernels[DT, DT, DT, False]()
    comptime cfg = kernels.ampere_128x128_4
    comptime BK = cfg.block_tile_shape[2]

    var ab = ctx.enqueue_create_buffer[DT](M * K)
    var bb = ctx.enqueue_create_buffer[DT](K * N)
    var cb = ctx.enqueue_create_buffer[DT](M * N)
    ab.enqueue_fill(Float32(0.01))
    bb.enqueue_fill(Float32(0.02))
    ctx.synchronize()
    var av = TileTensor(ab, row_major[M, K]())
    var bv = TileTensor(bb, row_major[K, N]())
    var cv = TileTensor(cb, row_major[M, N]())

    var best_p = 1
    var best_us = Float64(1e30)

    comptime for i in range(len(PS)):
        comptime P = PS[i]
        comptime PART = ((K // P) + BK - 1) // BK * BK
        comptime if (P - 1) * PART < K and P * PART >= K:
            if P * M * N <= ws.capacity:
                for _ in range(3):
                    splitk_gemm[transpose_b=False, config=cfg](
                        cv, av, bv, P, ws, ctx
                    )
                ctx.synchronize()
                var ta = perf_counter_ns()
                for _ in range(reps):
                    splitk_gemm[transpose_b=False, config=cfg](
                        cv, av, bv, P, ws, ctx
                    )
                ctx.synchronize()
                var us = Float64(perf_counter_ns() - ta) / 1000.0 / Float64(reps)
                if us < best_us:
                    best_us = us
                    best_p = P

    _ = ab^
    _ = bb^
    _ = cb^
    return best_p


def sweep_partitions[
    M: Int, K: Int, N: Int, LABEL: StaticString
](ctx: DeviceContext, mut ws: SplitKWorkspace[DT]) raises:
    """Sweep `num_k_partitions` past what `select_config` would choose.

    `select_config` caps P by `min_k_partition = 1024`: it will not cut K into
    pieces smaller than 1024, so at K=2592 it stops at P=2. That is a heuristic
    in a HOST-SIDE CHOOSER, not a constraint of the kernel -- the kernel takes
    the partition count as a runtime argument and MAX's own guard is only
    `K >= P * BK` (BK=16 here, so P=8 needs K>=128).

    Now that we launch the kernel ourselves, the cap is ours to pick. It is
    worth picking deliberately, because at these shapes the grid is tiny:
    [256 x 2592] @ [2592 x 256] with BM=BN=128 is 2x2 tiles, so P=2 puts
    EIGHT blocks on a 170-SM card. That is why the P=2 timings barely move
    between N=256 and N=1024 -- 4x the work fits in the same latency because
    the machine was idle either way. More partitions is more blocks.

    ⚠ Higher P is not free and not exact. Each partition is a separate fp32
    accumulation that the reduce then sums, so |A - B| stops being 0 as P
    grows -- that is arithmetic, not a bug, but it is a reason to choose P
    on evidence rather than maximising it.
    """
    comptime kernels = MatmulKernels[DT, DT, DT, False]()
    comptime cfg = kernels.ampere_128x128_4
    var picked = select_config[DT, DT, DT, False](M, N, K, ctx)
    if not (picked == cfg):
        print("  ", LABEL, ": select_config chose another tile; skipping", sep="")
        return

    var ab = ctx.enqueue_create_buffer[DT](M * K)
    var bb = ctx.enqueue_create_buffer[DT](K * N)
    var c_ref = ctx.enqueue_create_buffer[DT](M * N)
    var c_ours = ctx.enqueue_create_buffer[DT](M * N)
    ab.enqueue_fill(Float32(0.01))
    bb.enqueue_fill(Float32(0.02))
    ctx.synchronize()

    var av = TileTensor(ab, row_major[M, K]())
    var bv = TileTensor(bb, row_major[K, N]())
    var cref = TileTensor(c_ref, row_major[M, N]())
    var cours = TileTensor(c_ours, row_major[M, N]())

    for _ in range(WARMUP):
        max_matmul[target="gpu"](cref, av, bv, ctx)
    ctx.synchronize()

    var tiles = ceildiv(M, 128) * ceildiv(N, 128)
    comptime sm_count = ctx.default_device_info.sm_count
    var want = choose_partitions(
        M, N, K,
        cfg.block_tile_shape[0], cfg.block_tile_shape[1],
        cfg.block_tile_shape[2],
        sm_count,
    )
    var tuned = autotune_partitions[M, K, N](ctx, ws)
    print(
        "  ", LABEL, " [", M, "x", K, "] @ [", K, "x", N, "]  ",
        tiles, " tiles, ", sm_count, " SMs",
        "  select_config P=", picked.num_k_partitions,
        "  chooser P=", want, " (", tiles * want, " blocks)",
        "  autotuned P=", tuned, " (", tiles * tuned, " blocks)", sep="",
    )

    # Candidate partition counts. The overrun rule (see splitk_gemm) is
    # `(P-1) * align_up(K//P, BK) < K`; it is fully comptime here, so P values
    # that would fault are never instantiated rather than raised on.
    comptime BK = cfg.block_tile_shape[2]
    comptime for i in range(len(PS)):
        comptime P = PS[i]
        comptime PART = ((K // P) + BK - 1) // BK * BK
        comptime NO_OVERRUN = (P - 1) * PART < K
        comptime COVERS = P * PART >= K
        comptime FITS = NO_OVERRUN and COVERS
        comptime if not NO_OVERRUN:
            print(
                "      P=", P, "  SKIPPED (overrun): (P-1)*align_up(K//P,BK) = ",
                (P - 1) * PART, " >= K=", K,
                " -> the last block would read past the operands", sep="",
            )
        comptime if NO_OVERRUN and not COVERS:
            print(
                "      P=", P, "  SKIPPED (undercovers): P*align_up(K//P,BK) = ",
                P * PART, " < K=", K, " -> ", K - P * PART,
                " contraction elements silently dropped, rel err would be ",
                Float64(K - P * PART) / Float64(K), sep="",
            )
        comptime if FITS:
            if P * M * N <= ws.capacity:
                for _ in range(WARMUP):
                    splitk_gemm[transpose_b=False, config=cfg](
                        cours, av, bv, P, ws, ctx
                    )
                ctx.synchronize()
                var ta = perf_counter_ns()
                for _ in range(REPS):
                    splitk_gemm[transpose_b=False, config=cfg](
                        cours, av, bv, P, ws, ctx
                    )
                ctx.synchronize()
                var tb = perf_counter_ns()

                # Report RELATIVE error beside the absolute one. Each
                # partition is its own fp32 accumulation, so the difference
                # grows with P; 7e-7 means nothing until you know the entries
                # are ~0.5, at which point it is ~12 ulps and fine.
                var worst = Float64(0.0)
                var mag = Float64(0.0)
                with c_ref.map_to_host() as hr:
                    with c_ours.map_to_host() as ho:
                        for j in range(M * N):
                            var r = abs(Float64(hr[j]))
                            if r > mag:
                                mag = r
                            var d = abs(r - abs(Float64(ho[j])))
                            if d > worst:
                                worst = d

                print(
                    "      P=", P, "  blocks=", tiles * P,
                    "  ", Float64(tb - ta) / 1000.0 / Float64(REPS), "us",
                    "  |A-B| ", worst,
                    "  rel ", (worst / mag) if mag > 0.0 else 0.0, sep="",
                )

    _ = ab^
    _ = bb^
    _ = c_ref^
    _ = c_ours^


def sweep_pad[
    M: Int, K: Int, N_REAL: Int, N_PAD: Int, COUNT: Int, LABEL: StaticString
](ctx: DeviceContext, mut ws: SplitKWorkspace[DT]) raises:
    """Is it worth PADDING N to reach the multistage path?

    `Conv2D`'s dW is `[OC, BS] @ [BS, CPAD]`, and `CPAD` rounds the im2col
    column count to a multiple of 32 — the FORWARD's contraction alignment.
    For the dW GEMM that same number is N, and `multi_gemm_cond` wants
    `n % 128 == 0`, so a ResNet18 stem (CPAD=160) and every 64-input 3x3
    (CPAD=576) go to the VENDOR fallback instead. Those are the LONGEST-K
    GEMMs in the model (BS = 307,200 and 76,800), so they are the ones split-K
    would help most, and the ones it cannot currently touch.

    Padding N to 128 would admit them — at the cost of computing columns
    nobody reads: 160 -> 256 is 60% more FLOPs, 576 -> 640 is 11% more.

    The two numbers we had CONFLICTED and could not settle it:
    MODULAR_MATMUL_ALLOC_REPORT.md Measurement 1 has `[64 x 307200] @
    [307200 x 160]` at 483 us WITH nsys attached; this file's earlier sweep has
    the padded `[... x 256]` at 1706 us with NO profiler. Different conditions,
    not a comparison. This measures all three arms in one unprofiled process.

      A  max_matmul, N=N_REAL   the status quo (vendor fallback)
      B  max_matmul, N=N_PAD    multistage at MAX's own partition count
      C  splitk_gemm, N=N_PAD   ours, at the chooser's P
      F  the FORWARD at both widths — see below

    ⚠ THE dW IS ONLY HALF THE COST, and the first version of this measured
    only that half. `CPAD` is the im2col ROW STRIDE, so it is also the
    FORWARD's contraction: `out[BS, OCPAD] = col[BS, CPAD] @ w[CPAD, OCPAD]`.
    Widening it to admit the dW to multistage makes the FORWARD wider too, and
    the forward is the larger GEMM. For a ResNet18 stem that is +60% forward
    FLOPs against a dW win of 68 us/call — plausibly a NET LOSS. For layer1 it
    is +11% against 166 us/call, which is not close. So the verdict has to be
    computed on `forward_delta + dW_delta`, not on the dW alone.

    ⚠ CORRECTNESS IS NOT "EXACTLY EQUAL", and an earlier version of this
    docstring claimed it was. Column j of C reads only column j of B, so the
    two arms compute the same MATHEMATICAL result — but through DIFFERENT
    KERNELS (cuBLAS vs multistage split-K), hence different summation orders.
    At K = 307,200 a single fp32 sequential sum carries up to `K * eps` = 3.7e-2
    relative on its own, so a 1.5e-3 disagreement is ordering, not a defect.
    Same mistake as [[_an_exact_equality_dedup_needs_bit_identical_construction]]:
    the same maths through different arithmetic is not the same number.

    So this reports RELATIVE error against a float64 host reference, and scores
    BOTH arms against it — the question is not "do they agree" but "is the
    padded arm at least as accurate", which the reference can answer and a
    pairwise diff cannot.

    ⚠ AND THE ANSWER IS NO, FOR A REASON THAT IS NOT A DEFECT: **the two arms
    run at different PRECISION.** The vendor path takes `use_tf32 = False` by
    default and sets `CUBLAS_DEFAULT_MATH` (blas.mojo:391, 756-763) — full
    fp32. The multistage path CANNOT turn TF32 off for fp32 inputs outside
    SM100; `_matmul_gpu` asserts "use_tf32=False is only implemented for the
    SM100 matmul dispatch" (matmul/gpu/__init__.mojo:494). TF32 carries a
    10-bit mantissa, i.e. ~1e-3 relative — exactly the size of the gap
    measured (A 2.3e-3 vs C 4.7e-3 on the stem; 1.1e-3 vs 5.8e-3 on layer1).

    So padding does not merely reschedule these GEMMs, it moves them from fp32
    to TF32. Worth knowing before adopting it — though note this makes them
    CONSISTENT with the rest of the model rather than less accurate than it:
    every GEMM already on the multistage path is TF32, and the stem and
    64-input convs are only fp32 because they accidentally FAIL a shape gate.

    ⚠ Normalise by `sum |a_k * b_k|`, not by `|result|`. The fill is
    sign-changing, so the dot product cancels heavily and `|result|` can be
    orders of magnitude below the terms that built it; dividing by it inflates
    both arms' error and makes the numbers unquotable. The RATIO between the
    arms survives either way (same data, same denominator) but the absolute
    figures only mean something against the conditioned denominator.

    ⚠ It also fills with VARIED data. A constant fill makes every product
    identical, which removes the cancellation that exposes ordering defects and
    makes any summation look well conditioned.
    """
    var a = ctx.enqueue_create_buffer[DT](M * K)
    var bn = ctx.enqueue_create_buffer[DT](K * N_REAL)
    var bp = ctx.enqueue_create_buffer[DT](K * N_PAD)
    var cn = ctx.enqueue_create_buffer[DT](M * N_REAL)
    var cp = ctx.enqueue_create_buffer[DT](M * N_PAD)
    # Varied, sign-changing data: a constant fill makes every product equal,
    # which hides exactly the ordering effects this is meant to expose.
    var ah = ctx.enqueue_create_host_buffer[DT](M * K)
    var bh = ctx.enqueue_create_host_buffer[DT](K * N_PAD)
    ctx.synchronize()
    for i in range(M * K):
        ah[i] = Scalar[DT](0.02) * Scalar[DT]((i % 61) - 30)
    for i in range(K * N_PAD):
        bh[i] = Scalar[DT](0.01) * Scalar[DT]((i % 47) - 23)
    ctx.enqueue_copy(a, ah)
    ctx.enqueue_copy(bp, bh)
    # bn is bp's first N_REAL columns, row by row, so the two arms see the
    # SAME operand values and only the kernel differs.
    var bnh = ctx.enqueue_create_host_buffer[DT](K * N_REAL)
    ctx.synchronize()
    for i in range(K):
        for j in range(N_REAL):
            bnh[i * N_REAL + j] = bh[i * N_PAD + j]
    ctx.enqueue_copy(bn, bnh)
    ctx.synchronize()

    var av = TileTensor(a, row_major[M, K]())
    var bnv = TileTensor(bn, row_major[K, N_REAL]())
    var bpv = TileTensor(bp, row_major[K, N_PAD]())
    var cnv = TileTensor(cn, row_major[M, N_REAL]())
    var cpv = TileTensor(cp, row_major[M, N_PAD]())

    comptime kernels = MatmulKernels[DT, DT, DT, False]()
    comptime cfg = kernels.ampere_128x128_4
    comptime sm_count = ctx.default_device_info.sm_count
    var picked = select_config[DT, DT, DT, False](M, N_PAD, K, ctx)
    var want = choose_partitions(
        M, N_PAD, K,
        cfg.block_tile_shape[0], cfg.block_tile_shape[1],
        cfg.block_tile_shape[2], sm_count,
    )

    comptime R = 20

    for _ in range(3):
        max_matmul[target="gpu"](cnv, av, bnv, ctx)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(R):
        max_matmul[target="gpu"](cnv, av, bnv, ctx)
    ctx.synchronize()
    var t1 = perf_counter_ns()

    for _ in range(3):
        max_matmul[target="gpu"](cpv, av, bpv, ctx)
    ctx.synchronize()
    var t2 = perf_counter_ns()
    for _ in range(R):
        max_matmul[target="gpu"](cpv, av, bpv, ctx)
    ctx.synchronize()
    var t3 = perf_counter_ns()

    var us_c = Float64(0)
    var ok = picked == cfg and want > 1 and want * M * N_PAD <= ws.capacity
    if ok:
        for _ in range(3):
            splitk_gemm[transpose_b=False, config=cfg](
                cpv, av, bpv, want, ws, ctx
            )
        ctx.synchronize()
        var t4 = perf_counter_ns()
        for _ in range(R):
            splitk_gemm[transpose_b=False, config=cfg](
                cpv, av, bpv, want, ws, ctx
            )
        ctx.synchronize()
        us_c = Float64(perf_counter_ns() - t4) / 1000.0 / Float64(R)

    var us_a = Float64(t1 - t0) / 1000.0 / Float64(R)
    var us_b = Float64(t3 - t2) / 1000.0 / Float64(R)

    # ---- the forward, at both im2col widths ---------------------------
    # `out[BS, OCPAD] = col[BS, CPAD] @ w[CPAD, OCPAD]`. Padding CPAD widens
    # this GEMM's K, and it is the bigger of the two.
    comptime OCPAD = 128
    var colN = ctx.enqueue_create_buffer[DT](K * N_REAL)
    var colP = ctx.enqueue_create_buffer[DT](K * N_PAD)
    var wN = ctx.enqueue_create_buffer[DT](N_REAL * OCPAD)
    var wP = ctx.enqueue_create_buffer[DT](N_PAD * OCPAD)
    var outf = ctx.enqueue_create_buffer[DT](K * OCPAD)
    colN.enqueue_fill(Float32(0.01))
    colP.enqueue_fill(Float32(0.01))
    wN.enqueue_fill(Float32(0.02))
    wP.enqueue_fill(Float32(0.02))
    ctx.synchronize()
    var colNv = TileTensor(colN, row_major[K, N_REAL]())
    var colPv = TileTensor(colP, row_major[K, N_PAD]())
    var wNv = TileTensor(wN, row_major[N_REAL, OCPAD]())
    var wPv = TileTensor(wP, row_major[N_PAD, OCPAD]())
    var outfv = TileTensor(outf, row_major[K, OCPAD]())

    for _ in range(3):
        max_matmul[target="gpu"](outfv, colNv, wNv, ctx)
    ctx.synchronize()
    var f0 = perf_counter_ns()
    for _ in range(R):
        max_matmul[target="gpu"](outfv, colNv, wNv, ctx)
    ctx.synchronize()
    var f1 = perf_counter_ns()
    for _ in range(3):
        max_matmul[target="gpu"](outfv, colPv, wPv, ctx)
    ctx.synchronize()
    var f2 = perf_counter_ns()
    for _ in range(R):
        max_matmul[target="gpu"](outfv, colPv, wPv, ctx)
    ctx.synchronize()
    var f3 = perf_counter_ns()
    var fwd_n = Float64(f1 - f0) / 1000.0 / Float64(R)
    var fwd_p = Float64(f3 - f2) / 1000.0 / Float64(R)

    # Score BOTH arms against a float64 host reference. Checking one against
    # the other only says they differ, which they must -- different kernels,
    # different summation order. The reference says WHICH is closer to the
    # truth, which is the question worth asking. Sampled columns: a full
    # reference at K=307200 would dominate the run time.
    comptime NCHK = 8
    var ref_a = Float64(0)
    var ref_c = Float64(0)
    with cn.map_to_host() as hn:
        with cp.map_to_host() as hp:
            for jj in range(NCHK):
                var j = (jj * N_REAL) // NCHK
                var acc = Float64(0)
                var cond = Float64(0)
                for k in range(K):
                    var t = Float64(ah[k]) * Float64(bh[k * N_PAD + j])
                    acc += t
                    cond += abs(t)
                # Condition-aware denominator: `sum |a_k b_k|`, not |acc|.
                # With sign-changing data |acc| can sit far below the terms
                # that produced it, and dividing by it reports the dot
                # product's CONDITION NUMBER rather than the kernel's error.
                var mag = cond + 1e-30
                var da = abs(Float64(hn[j]) - acc) / mag
                var dc = abs(Float64(hp[j]) - acc) / mag
                if da > ref_a:
                    ref_a = da
                if dc > ref_c:
                    ref_c = dc

    print("  ", LABEL, "  [", M, " x ", K, "] @ [", K, " x N]   x", COUNT,
          "/step", sep="")
    print("      A  N=", N_REAL, " max_matmul (vendor)      ", us_a, "us", sep="")
    print("      B  N=", N_PAD, " max_matmul (multistage P=",
          picked.num_k_partitions, ")  ", us_b, "us   ", us_a / us_b, "x vs A",
          sep="")
    if ok:
        print("      C  N=", N_PAD, " splitk_gemm P=", want, "          ",
              us_c, "us   ", us_a / us_c, "x vs A", sep="")
        var d_dw = us_a - us_c            # positive = padding helps
        var d_fw = fwd_n - fwd_p          # negative = padding costs
        var net = d_dw + d_fw
        print("      F  forward [", K, " x N] @ [N x ", OCPAD, "]:  N=", N_REAL,
              " ", fwd_n, "us   N=", N_PAD, " ", fwd_p, "us   delta ", -d_fw,
              "us", sep="")
        print("         NET per call: dW ", -d_dw, " + fwd ", -d_fw, " = ",
              -net, "us   -> ",
              "PAD WINS" if net > 0.0 else "PAD LOSES — keep the vendor path",
              sep="")
        print("         net per step at x", COUNT, ": ",
              net * Float64(COUNT) / 1000.0, " ms", sep="")
    else:
        print("      C  skipped (chooser P=", want, ", tile match=",
              picked == cfg, ")", sep="")
    print("      err vs float64 (normalised by sum|a.b|):  A(vendor, fp32) ",
          ref_a, "   C(split-K padded, TF32) ", ref_c, sep="")
    print("         ratio C/A = ", ref_c / (ref_a + 1e-30),
          "x — expected > 1: the vendor path runs FULL fp32"
          " (CUBLAS_DEFAULT_MATH) and multistage CANNOT disable TF32 for fp32"
          " outside SM100. This is a precision CHANGE, not a defect.", sep="")

    _ = a^
    _ = bn^
    _ = bp^
    _ = cn^
    _ = cp^
    _ = ah^
    _ = bh^
    _ = bnh^
    _ = colN^
    _ = colP^
    _ = wN^
    _ = wP^
    _ = outf^


def main() raises:
    comptime if not default_accelerator().is_nvidia_gpu():
        print("NVIDIA only -- build with `pixi run -e nvidia`.")
    else:
        with DeviceContext() as ctx:
            # 8 partitions is select_config's ceiling; the widest M*N below is
            # 1024*256. Sized once, before anything that would be captured.
            var ws = SplitKWorkspace[DT](ctx, 8 * 1024 * 1024)

            var total = Float64(0.0)
            print("ACT dW shapes, BATCH=16 DIM=256 FF=1024 K=60 ENC=4 DEC=1")
            print("dW GEMM is [in_features x B] @ [B x out_features], B = batch*tokens")
            print()
            print("-- transformer encoder, B = 16*162 = 2592 --------------------")
            # 4 layers x {q,k,v,ao} = 16, plus the decoder's cross-attention
            # k and v, which run over the 162 MEMORY tokens, not the 60 queries.
            sweep_one[256, 2592, 256, 18, "enc attn qkv/out + dec cross k,v"](ctx, ws, total)
            sweep_one[256, 2592, 1024, 4, "enc ff1                        "](ctx, ws, total)
            sweep_one[1024, 2592, 256, 4, "enc ff2                        "](ctx, ws, total)

            print()
            print("-- controls: these should NOT split (B < 2048) ---------------")
            sweep_one[256, 992, 256, 16, "CVAE enc attn, B=16*62=992     "](ctx, ws, total)
            sweep_one[256, 960, 256, 6, "decoder self/cross-q, B=960    "](ctx, ws, total)
            sweep_one[256, 960, 1024, 1, "decoder ff1                    "](ctx, ws, total)

            print()
            print("-- conv dW family (M=out_ch, N=col, K=images*OH*OW) ----------")
            # MEASURED in MODULAR_MATMUL_ALLOC_REPORT.md Measurement 4.
            sweep_one[128, 32768, 128, 1, "measured in the report         "](ctx, ws, total)
            # ResNet18 stem dW, N padded 160 -> 256 to clear `n % 128`.
            sweep_one[64, 307200, 256, 2, "ResNet18 stem dW (N padded)    "](ctx, ws, total)

            print()
            print("TOTAL saved per training step (listed shapes only):", total, "us")
            print(
                "  = ", total / 1000.0, " ms/step; over 1000 steps, ",
                total / 1000.0, " s", sep="",
            )
            print()
            print(
                "⚠ Counts are from the layer list, not from a run. Confirm with"
                " the LOGGING_LEVEL=INFO command in this file's docstring"
                " before quoting the total."
            )

            print()
            print("-- P sweep: select_config's cap is a heuristic, not a limit --")
            print("   (grid is tiny at these shapes; more partitions = more SMs)")
            sweep_partitions[256, 2592, 256, "enc attn      "](ctx, ws)
            sweep_partitions[256, 2592, 1024, "enc ff1       "](ctx, ws)
            sweep_partitions[1024, 2592, 256, "enc ff2       "](ctx, ws)
            print()
            print("-- would padding N to 128 admit the conv dW? ------------")
            print("   (stem and the 64-input 3x3s are the LONGEST K in ACT")
            print("    and the only convs multi_gemm_cond excludes)")
            # ResNet18 stem dW: COL=147 -> CPAD=160, BS=16*120*160.
            sweep_pad[64, 307200, 160, 256, 2, "stem 7x7  "](ctx, ws)
            # layer1 3x3 dW: COL=576 -> CPAD=576, BS=16*60*80.
            sweep_pad[64, 76800, 576, 640, 4, "layer1 3x3"](ctx, ws)
            _ = ws^
