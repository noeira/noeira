"""BatchNorm2D — NCHW vs NHWC parity (perf + correctness). [BN-only split]

Split out of the former bench_bn_pool_nhwc_parity (BN + Pool in one module) to cut
compile time + codegen pressure (the combined file's 4 BN variants × 3 shapes ×
comptime-unrolled timing loops SIGILLed the Metal backend on the 2D kernel). Pool
parity lives in bench_pool_nhwc_parity_gpu.mojo.

BN reduction is over (N, spatial) PER CHANNEL:
  • NCHW: spatial contiguous within a channel → block-per-channel coalesces. The
    REAL BatchNorm2D is G-grouped (grid=C*G, ~2-4k blocks) → the honest baseline;
    the strawman (grid=C, 32-64 blocks) is GPU-starved and inflates the NHWC ratio.
  • NHWC: channel contiguous, spatial strided → TRANSPOSED reduction. 1-warp form
    (block_dim=C) is thread-starved; the 2D form (block_dim=ROWS×C, per-rowgroup
    partials + parallel block.sum finalize) is the occupancy fix.

Four BN variants timed per shape: NCHW-strawman, NCHW-real(G-grouped), NHWC-1warp,
NHWC-2D. The 2D is NVIDIA-gated in this bench (its high-occupancy launch SIGILLs
Metal *here*; in a minimal module it's Metal-safe — see bench_bn_nhwc_2d_isolated).

Run (NVIDIA = perf truth):
    pixi run -e nvidia mojo run -I . benchmarks/bench_bn_nhwc_parity_gpu.mojo
Run (Apple = parity only; 2D skipped):
    pixi run -e apple  mojo run -I . benchmarks/bench_bn_nhwc_parity_gpu.mojo
"""

from std.math import sqrt
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext, DeviceBuffer
from std.sys.info import default_accelerator
from std.time import perf_counter_ns
from noeira.core.tiles import RowMajor, Tile, tile

comptime DT = DType.float32
comptime TPB = 128        # block-per-channel reduction width (NCHW) / flat launches
comptime NCHUNK = 256     # NHWC partial-reduction row chunks. 1024 was WORSE than
                          # 512 (more chunks → more partials for the grid=C finalize
                          # to reduce — the FINALIZE is the bottleneck, not the
                          # partial). Testing 256: partial still 256*256≈65k threads
                          # (plenty), finalize does half the work of 512.
comptime BN2D_BLK = 256   # NHWC-2D reduction block (= ROWS_PER_BLK * C; ROWS = BLK/C)
comptime EPS = 1e-5


# ══════════════════════════════════════════════════════════════════════════
# BatchNorm2D — NCHW strawman (grid=C): block per channel, threads stride spatial
# ══════════════════════════════════════════════════════════════════════════
def _bn_stats_nchw[
    N: Int, C: Int, SP: Int, FLAT: Int,
](
    input: Tile[DT, N, FLAT],
    mean_out: Tile[DT, C],
    inv_std_out: Tile[DT, C],
):
    var c = Int(block_idx.x)
    if c >= C:
        return
    var t = Int(thread_idx.x)
    var c_off = c * SP
    var my_sum: Scalar[DT] = 0.0
    var my_sumsq: Scalar[DT] = 0.0
    for b in range(N):
        var s = t
        while s < SP:
            var x = rebind[Scalar[DT]](input[b, c_off + s])
            my_sum += x
            my_sumsq += x * x
            s += TPB
    var bsum = block.sum[block_size=TPB, broadcast=False](val=my_sum)
    var bsq = block.sum[block_size=TPB, broadcast=False](val=my_sumsq)
    if t == 0:
        var inv_n = Scalar[DT](1.0) / Scalar[DT](Float32(N * SP))
        var mean = bsum[0] * inv_n
        var var_ = bsq[0] * inv_n - mean * mean
        if var_ < Scalar[DT](0.0):
            var_ = Scalar[DT](0.0)
        mean_out[c] = mean
        inv_std_out[c] = Scalar[DT](1.0) / sqrt(var_ + Scalar[DT](EPS))


# ── REAL NCHW design: G-grouped partial + finalize (matches batch_norm_2d.mojo) ─
# grid = C*G (≈2-4k blocks), coalesced AND well-parallelized → the honest baseline.
def _bn_stats_nchw_grouped[
    N: Int, C: Int, SP: Int, FLAT: Int, G: Int,
](
    input: Tile[DT, N, FLAT],
    partial_sum: Tile[DT, C * G],
    partial_sumsq: Tile[DT, C * G],
):
    var blk = Int(block_idx.x)
    var c = blk // G
    var g = blk % G
    if c >= C:
        return
    var t = Int(thread_idx.x)
    var bpb = (N + G - 1) // G
    var b0 = g * bpb
    var b1 = b0 + bpb
    if b1 > N:
        b1 = N
    var c_off = c * SP
    var my_sum: Scalar[DT] = 0.0
    var my_sumsq: Scalar[DT] = 0.0
    for b in range(b0, b1):
        var s = t
        while s < SP:
            var x = rebind[Scalar[DT]](input[b, c_off + s])
            my_sum += x
            my_sumsq += x * x
            s += TPB
    var bsum = block.sum[block_size=TPB, broadcast=False](val=my_sum)
    var bsq = block.sum[block_size=TPB, broadcast=False](val=my_sumsq)
    if t == 0:
        partial_sum[c * G + g] = bsum[0]
        partial_sumsq[c * G + g] = bsq[0]


def _bn_finalize_nchw_grouped[
    C: Int, G: Int, RTOT: Int,
](
    partial_sum: Tile[DT, C * G],
    partial_sumsq: Tile[DT, C * G],
    mean_out: Tile[DT, C],
    inv_std_out: Tile[DT, C],
):
    var c = Int(global_idx.x)
    if c >= C:
        return
    var s: Scalar[DT] = 0.0
    var sq: Scalar[DT] = 0.0
    for g in range(G):
        s += rebind[Scalar[DT]](partial_sum[c * G + g])
        sq += rebind[Scalar[DT]](partial_sumsq[c * G + g])
    var inv_n = Scalar[DT](1.0) / Scalar[DT](Float32(RTOT))
    var mean = s * inv_n
    var var_ = sq * inv_n - mean * mean
    if var_ < Scalar[DT](0.0):
        var_ = Scalar[DT](0.0)
    mean_out[c] = mean
    inv_std_out[c] = Scalar[DT](1.0) / sqrt(var_ + Scalar[DT](EPS))


# normalize: strawman (grid=C) + G-grouped (grid=C*G).
def _bn_norm_nchw[
    N: Int, C: Int, SP: Int, FLAT: Int,
](
    input: Tile[DT, N, FLAT],
    output: Tile[DT, N, FLAT],
    gamma: Tile[DT, C],
    beta: Tile[DT, C],
    mean_in: Tile[DT, C],
    inv_std_in: Tile[DT, C],
):
    var c = Int(block_idx.x)
    if c >= C:
        return
    var t = Int(thread_idx.x)
    var c_off = c * SP
    var mean = rebind[Scalar[DT]](mean_in[c])
    var inv_std = rebind[Scalar[DT]](inv_std_in[c])
    var gm = rebind[Scalar[DT]](gamma[c])
    var bt = rebind[Scalar[DT]](beta[c])
    for b in range(N):
        var s = t
        while s < SP:
            var off = c_off + s
            var xh = (rebind[Scalar[DT]](input[b, off]) - mean) * inv_std
            output[b, off] = gm * xh + bt
            s += TPB


def _bn_norm_nchw_grouped[
    N: Int, C: Int, SP: Int, FLAT: Int, G: Int,
](
    input: Tile[DT, N, FLAT],
    output: Tile[DT, N, FLAT],
    gamma: Tile[DT, C],
    beta: Tile[DT, C],
    mean_in: Tile[DT, C],
    inv_std_in: Tile[DT, C],
):
    var blk = Int(block_idx.x)
    var c = blk // G
    var g = blk % G
    if c >= C:
        return
    var t = Int(thread_idx.x)
    var bpb = (N + G - 1) // G
    var b0 = g * bpb
    var b1 = b0 + bpb
    if b1 > N:
        b1 = N
    var c_off = c * SP
    var mean = rebind[Scalar[DT]](mean_in[c])
    var inv_std = rebind[Scalar[DT]](inv_std_in[c])
    var gm = rebind[Scalar[DT]](gamma[c])
    var bt = rebind[Scalar[DT]](beta[c])
    for b in range(b0, b1):
        var s = t
        while s < SP:
            var off = c_off + s
            var xh = (rebind[Scalar[DT]](input[b, off]) - mean) * inv_std
            output[b, off] = gm * xh + bt
            s += TPB


# ══════════════════════════════════════════════════════════════════════════
# BatchNorm2D — NHWC (channels-last): TRANSPOSED reduction
# ══════════════════════════════════════════════════════════════════════════
# 1-warp partial: grid=NCHUNK, block_dim=C (one thread per channel, coalesced but
# thread-starved). The honest 1-warp baseline.
def _bn_stats_nhwc_partial[
    C: Int, R: Int,
](
    input: Tile[DT, R * C],
    partial_sum: Tile[DT, NCHUNK * C],
    partial_sumsq: Tile[DT, NCHUNK * C],
):
    var chunk = Int(block_idx.x)
    var c = Int(thread_idx.x)
    if c >= C:
        return
    comptime rpc = (R + NCHUNK - 1) // NCHUNK
    var r0 = chunk * rpc
    if r0 >= R:
        partial_sum[chunk * C + c] = Scalar[DT](0.0)
        partial_sumsq[chunk * C + c] = Scalar[DT](0.0)
        return
    var r1 = r0 + rpc
    if r1 > R:
        r1 = R
    var my_sum: Scalar[DT] = 0.0
    var my_sumsq: Scalar[DT] = 0.0
    for r in range(r0, r1):
        var x = rebind[Scalar[DT]](input[r * C + c])
        my_sum += x
        my_sumsq += x * x
    partial_sum[chunk * C + c] = my_sum
    partial_sumsq[chunk * C + c] = my_sumsq


# 2D partial: OCCUPANCY fix. block_dim = BN2D_BLK = ROWS row-groups × C channels.
# Thread (rg,c) accumulates its channel over a strided row subset and writes its
# OWN partial at (chunk*ROWS+rg)*C+c (no shared mem) → 8 warps/block (C=32) = full
# occupancy. Cross-rowgroup reduction deferred to the parallel finalize.
def _bn_stats_nhwc_2d[
    C: Int, R: Int, CHUNKS: Int,
](
    input: Tile[DT, R * C],
    partial_sum: Tile[DT, CHUNKS * BN2D_BLK],
    partial_sumsq: Tile[DT, CHUNKS * BN2D_BLK],
):
    comptime ROWS = BN2D_BLK // C
    var chunk = Int(block_idx.x)
    var lane = Int(thread_idx.x)
    var c = lane % C
    var rg = lane // C
    comptime rpc = (R + CHUNKS - 1) // CHUNKS
    var r0 = chunk * rpc
    var r1 = r0 + rpc
    if r1 > R:
        r1 = R
    var my_sum: Scalar[DT] = 0.0
    var my_sq: Scalar[DT] = 0.0
    var r = r0 + rg
    while r < r1:
        var x = rebind[Scalar[DT]](input[r * C + c])
        my_sum += x
        my_sq += x * x
        r += ROWS
    var pidx = (chunk * ROWS + rg) * C + c
    partial_sum[pidx] = my_sum
    partial_sumsq[pidx] = my_sq


# Parallel finalize: grid=C, block reduces its P = CHUNKS*ROWS partials via block.sum.
def _bn_finalize_nhwc_parallel[
    C: Int, RTOT: Int, P: Int,
](
    partial_sum: Tile[DT, P * C],
    partial_sumsq: Tile[DT, P * C],
    mean_out: Tile[DT, C],
    inv_std_out: Tile[DT, C],
):
    var c = Int(block_idx.x)
    if c >= C:
        return
    var t = Int(thread_idx.x)
    var my_s: Scalar[DT] = 0.0
    var my_sq: Scalar[DT] = 0.0
    var k = t
    while k < P:
        my_s += rebind[Scalar[DT]](partial_sum[k * C + c])
        my_sq += rebind[Scalar[DT]](partial_sumsq[k * C + c])
        k += TPB
    var bs = block.sum[block_size=TPB, broadcast=False](val=my_s)
    var bsq = block.sum[block_size=TPB, broadcast=False](val=my_sq)
    if t == 0:
        var inv_n = Scalar[DT](1.0) / Scalar[DT](Float32(RTOT))
        var mean = bs[0] * inv_n
        var var_ = bsq[0] * inv_n - mean * mean
        if var_ < Scalar[DT](0.0):
            var_ = Scalar[DT](0.0)
        mean_out[c] = mean
        inv_std_out[c] = Scalar[DT](1.0) / sqrt(var_ + Scalar[DT](EPS))


# 1-warp finalize: one thread per channel, sum the NCHUNK partials.
def _bn_finalize_nhwc[
    C: Int, RTOT: Int,
](
    partial_sum: Tile[DT, NCHUNK * C],
    partial_sumsq: Tile[DT, NCHUNK * C],
    mean_out: Tile[DT, C],
    inv_std_out: Tile[DT, C],
):
    var c = Int(global_idx.x)
    if c >= C:
        return
    var s: Scalar[DT] = 0.0
    var sq: Scalar[DT] = 0.0
    for k in range(NCHUNK):
        s += rebind[Scalar[DT]](partial_sum[k * C + c])
        sq += rebind[Scalar[DT]](partial_sumsq[k * C + c])
    var inv_n = Scalar[DT](1.0) / Scalar[DT](Float32(RTOT))
    var mean = s * inv_n
    var var_ = sq * inv_n - mean * mean
    if var_ < Scalar[DT](0.0):
        var_ = Scalar[DT](0.0)
    mean_out[c] = mean
    inv_std_out[c] = Scalar[DT](1.0) / sqrt(var_ + Scalar[DT](EPS))


# normalize: flat thread per element (R*C); c = idx % C → coalesced.
def _bn_norm_nhwc[
    C: Int, RC: Int,
](
    input: Tile[DT, RC],
    output: Tile[DT, RC],
    gamma: Tile[DT, C],
    beta: Tile[DT, C],
    mean_in: Tile[DT, C],
    inv_std_in: Tile[DT, C],
):
    var idx = Int(global_idx.x)
    if idx >= RC:
        return
    var c = idx % C
    var xh = (rebind[Scalar[DT]](input[idx]) - rebind[Scalar[DT]](mean_in[c])) \
        * rebind[Scalar[DT]](inv_std_in[c])
    output[idx] = rebind[Scalar[DT]](gamma[c]) * xh + rebind[Scalar[DT]](beta[c])


# ══════════════════════════════════════════════════════════════════════════
# host helper — fill one logical tensor into NCHW + NHWC buffers
# ══════════════════════════════════════════════════════════════════════════
def _fill_layouts[
    N: Int, C: Int, H: Int, W: Int, FLAT: Int,
](
    ctx: DeviceContext,
    nchw: DeviceBuffer[DT],
    nhwc: DeviceBuffer[DT],
) raises:
    var SP = H * W
    with nchw.map_to_host() as hn:
        with nhwc.map_to_host() as hh:
            for n in range(N):
                for c in range(C):
                    for s in range(SP):
                        var v = Scalar[DT](
                            Float64((((n * C + c) * SP + s) % 1009) + 1) * 0.03
                            - 15.0
                        )
                        hn[n * FLAT + c * SP + s] = v          # NCHW
                        hh[(n * SP + s) * C + c] = v           # NHWC


# ══════════════════════════════════════════════════════════════════════════
# BatchNorm: run all variants, verify, time
# ══════════════════════════════════════════════════════════════════════════
def run_bn[
    N: Int, C: Int, H: Int, W: Int, WARMUP: Int, ITERS: Int,
](ctx: DeviceContext, label: StaticString) raises:
    comptime SP = H * W
    comptime FLAT = C * SP
    comptime R = N * SP
    comptime RC = R * C
    print(label, " BN  N=", N, " C=", C, " ", H, "x", W, " | R=", R)

    var x_nchw = ctx.enqueue_create_buffer[DT](N * FLAT)
    var x_nhwc = ctx.enqueue_create_buffer[DT](RC)
    _fill_layouts[N, C, H, W, FLAT](ctx, x_nchw, x_nhwc)
    var out_nchw = ctx.enqueue_create_buffer[DT](N * FLAT)
    var out_nhwc = ctx.enqueue_create_buffer[DT](RC)
    var gamma = ctx.enqueue_create_buffer[DT](C)
    var beta = ctx.enqueue_create_buffer[DT](C)
    _ = gamma.enqueue_fill(Scalar[DT](1.3))
    _ = beta.enqueue_fill(Scalar[DT](-0.4))
    var mean_a = ctx.enqueue_create_buffer[DT](C)
    var istd_a = ctx.enqueue_create_buffer[DT](C)
    var mean_b = ctx.enqueue_create_buffer[DT](C)
    var istd_b = ctx.enqueue_create_buffer[DT](C)
    var ps = ctx.enqueue_create_buffer[DT](NCHUNK * BN2D_BLK)
    var psq = ctx.enqueue_create_buffer[DT](NCHUNK * BN2D_BLK)

    var gl = tile[C](gamma)
    var bl = tile[C](beta)

    var xn = tile[N, FLAT](x_nchw)
    var on = tile[N, FLAT](out_nchw)
    var ma = tile[C](mean_a)
    var ia = tile[C](istd_a)
    ctx.enqueue_function[_bn_stats_nchw[N, C, SP, FLAT]](
        xn, ma, ia, grid_dim=C, block_dim=TPB)
    ctx.enqueue_function[_bn_norm_nchw[N, C, SP, FLAT]](
        xn, on, gl, bl, ma, ia, grid_dim=C, block_dim=TPB)

    var xh = tile[RC](x_nhwc)
    var oh = tile[RC](out_nhwc)
    var mb = tile[C](mean_b)
    var ib = tile[C](istd_b)
    var psl = tile[NCHUNK * C](ps)
    var pql = tile[NCHUNK * C](psq)
    comptime ROWS = BN2D_BLK // C
    comptime P2 = NCHUNK * ROWS
    var psl2 = tile[NCHUNK * BN2D_BLK](ps)
    var pql2 = tile[NCHUNK * BN2D_BLK](psq)
    var xhr = tile[R * C](x_nhwc)
    comptime nb_norm = (RC + TPB - 1) // TPB
    ctx.enqueue_function[_bn_stats_nhwc_partial[C, R]](
        xhr, psl, pql, grid_dim=NCHUNK, block_dim=C)
    ctx.enqueue_function[_bn_finalize_nhwc[C, R]](
        psl, pql, mb, ib, grid_dim=(C + TPB - 1) // TPB, block_dim=TPB)
    ctx.enqueue_function[_bn_norm_nhwc[C, RC]](
        xh, oh, gl, bl, mb, ib, grid_dim=nb_norm, block_dim=TPB)
    ctx.synchronize()

    # ---- verify: stats + normalized outputs agree ----
    var stat_bad = 0
    with mean_a.map_to_host() as hma:
        with mean_b.map_to_host() as hmb:
            with istd_a.map_to_host() as hia:
                with istd_b.map_to_host() as hib:
                    for c in range(C):
                        if abs(Float64(hma[c] - hmb[c])) > 1e-2:
                            stat_bad += 1
                        if abs(Float64(hia[c] - hib[c])) > 1e-2:
                            stat_bad += 1
    var out_bad = 0
    with out_nchw.map_to_host() as hon:
        with out_nhwc.map_to_host() as hoh:
            for n in range(N):
                for c in range(C):
                    for s in range(SP):
                        var a = hon[n * FLAT + c * SP + s]
                        var bb = hoh[(n * SP + s) * C + c]
                        if abs(Float64(a - bb)) > 1e-2:
                            out_bad += 1
    print("  verify: stat_mismatch=", stat_bad, " out_mismatch=", out_bad)

    # ---- verify NHWC-2D stats vs NCHW ref (NVIDIA-only; Metal ICEs it here) ----
    var stat_bad2 = 0
    comptime if default_accelerator().is_nvidia_gpu():
        ctx.enqueue_function[_bn_stats_nhwc_2d[C, R, NCHUNK]](
            xhr, psl2, pql2, grid_dim=NCHUNK, block_dim=BN2D_BLK)
        ctx.enqueue_function[_bn_finalize_nhwc_parallel[C, R, P2]](
            psl2, pql2, mb, ib, grid_dim=C, block_dim=TPB)
        ctx.synchronize()
        with mean_a.map_to_host() as hma2:
            with mean_b.map_to_host() as hmb2:
                for c in range(C):
                    if abs(Float64(hma2[c] - hmb2[c])) > 1e-2:
                        stat_bad2 += 1
        print("  verify NHWC-2D: stat_mismatch=", stat_bad2)
    else:
        print("  verify NHWC-2D: skipped (NVIDIA-only — Metal ICEs the 2D kernel)")

    # ---- time NCHW strawman ----
    comptime for _ in range(WARMUP):
        ctx.enqueue_function[_bn_stats_nchw[N, C, SP, FLAT]](
            xn, ma, ia, grid_dim=C, block_dim=TPB)
        ctx.enqueue_function[_bn_norm_nchw[N, C, SP, FLAT]](
            xn, on, gl, bl, ma, ia, grid_dim=C, block_dim=TPB)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    comptime for _ in range(ITERS):
        ctx.enqueue_function[_bn_stats_nchw[N, C, SP, FLAT]](
            xn, ma, ia, grid_dim=C, block_dim=TPB)
        ctx.enqueue_function[_bn_norm_nchw[N, C, SP, FLAT]](
            xn, on, gl, bl, ma, ia, grid_dim=C, block_dim=TPB)
    ctx.synchronize()
    var us_nchw = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0

    # ---- time NHWC 1-warp ----
    comptime for _ in range(WARMUP):
        ctx.enqueue_function[_bn_stats_nhwc_partial[C, R]](
            xhr, psl, pql, grid_dim=NCHUNK, block_dim=C)
        ctx.enqueue_function[_bn_finalize_nhwc[C, R]](
            psl, pql, mb, ib, grid_dim=(C + TPB - 1) // TPB, block_dim=TPB)
        ctx.enqueue_function[_bn_norm_nhwc[C, RC]](
            xh, oh, gl, bl, mb, ib, grid_dim=nb_norm, block_dim=TPB)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    comptime for _ in range(ITERS):
        ctx.enqueue_function[_bn_stats_nhwc_partial[C, R]](
            xhr, psl, pql, grid_dim=NCHUNK, block_dim=C)
        ctx.enqueue_function[_bn_finalize_nhwc[C, R]](
            psl, pql, mb, ib, grid_dim=(C + TPB - 1) // TPB, block_dim=TPB)
        ctx.enqueue_function[_bn_norm_nhwc[C, RC]](
            xh, oh, gl, bl, mb, ib, grid_dim=nb_norm, block_dim=TPB)
    ctx.synchronize()
    var us_nhwc = Float64(perf_counter_ns() - t1) / Float64(ITERS) / 1000.0

    # ---- time NHWC-2D (NVIDIA) ----
    var us_nhwc2 = us_nhwc   # default on Apple (2D skipped)
    comptime if default_accelerator().is_nvidia_gpu():
        comptime for _ in range(WARMUP):
            ctx.enqueue_function[_bn_stats_nhwc_2d[C, R, NCHUNK]](
                xhr, psl2, pql2, grid_dim=NCHUNK, block_dim=BN2D_BLK)
            ctx.enqueue_function[_bn_finalize_nhwc_parallel[C, R, P2]](
                psl2, pql2, mb, ib, grid_dim=C, block_dim=TPB)
            ctx.enqueue_function[_bn_norm_nhwc[C, RC]](
                xh, oh, gl, bl, mb, ib, grid_dim=nb_norm, block_dim=TPB)
        ctx.synchronize()
        var t12 = perf_counter_ns()
        comptime for _ in range(ITERS):
            ctx.enqueue_function[_bn_stats_nhwc_2d[C, R, NCHUNK]](
                xhr, psl2, pql2, grid_dim=NCHUNK, block_dim=BN2D_BLK)
            ctx.enqueue_function[_bn_finalize_nhwc_parallel[C, R, P2]](
                psl2, pql2, mb, ib, grid_dim=C, block_dim=TPB)
            ctx.enqueue_function[_bn_norm_nhwc[C, RC]](
                xh, oh, gl, bl, mb, ib, grid_dim=nb_norm, block_dim=TPB)
        ctx.synchronize()
        us_nhwc2 = Float64(perf_counter_ns() - t12) / Float64(ITERS) / 1000.0

    # ---- time NCHW-grouped (the REAL BatchNorm2D design) ----
    comptime G = N if N < 64 else 64
    comptime lcg = RowMajor[C * G]
    var psg = tile[C * G](ps)
    var pqg = tile[C * G](psq)
    comptime nb_fin = (C + TPB - 1) // TPB
    comptime for _ in range(WARMUP):
        ctx.enqueue_function[_bn_stats_nchw_grouped[N, C, SP, FLAT, G]](
            xn, psg, pqg, grid_dim=C * G, block_dim=TPB)
        ctx.enqueue_function[_bn_finalize_nchw_grouped[C, G, R]](
            psg, pqg, ma, ia, grid_dim=nb_fin, block_dim=TPB)
        ctx.enqueue_function[_bn_norm_nchw_grouped[N, C, SP, FLAT, G]](
            xn, on, gl, bl, ma, ia, grid_dim=C * G, block_dim=TPB)
    ctx.synchronize()
    var t2 = perf_counter_ns()
    comptime for _ in range(ITERS):
        ctx.enqueue_function[_bn_stats_nchw_grouped[N, C, SP, FLAT, G]](
            xn, psg, pqg, grid_dim=C * G, block_dim=TPB)
        ctx.enqueue_function[_bn_finalize_nchw_grouped[C, G, R]](
            psg, pqg, ma, ia, grid_dim=nb_fin, block_dim=TPB)
        ctx.enqueue_function[_bn_norm_nchw_grouped[N, C, SP, FLAT, G]](
            xn, on, gl, bl, ma, ia, grid_dim=C * G, block_dim=TPB)
    ctx.synchronize()
    var us_nchw_g = Float64(perf_counter_ns() - t2) / Float64(ITERS) / 1000.0

    var gb = 2.0 * Float64(RC) * 4.0 / 1e9   # stream in+out once (norm pass)
    print("  BN NCHW strawman(grid=C):", us_nchw, "us ",
          gb / (us_nchw / 1e6) / 1e3, "TB/s")
    print("  BN NCHW real(G-grouped): ", us_nchw_g, "us ",
          gb / (us_nchw_g / 1e6) / 1e3, "TB/s  <- HONEST baseline")
    print("  BN NHWC transposed(1warp):", us_nhwc, "us ",
          gb / (us_nhwc / 1e6) / 1e3, "TB/s")
    print("  BN NHWC-2D (occupancy fix):", us_nhwc2, "us ",
          gb / (us_nhwc2 / 1e6) / 1e3, "TB/s")
    print("  >> NHWC-2D/realNCHW =", us_nhwc2 / us_nchw_g,
          "x  | NHWC-2D/1warp =", us_nhwc2 / us_nhwc, "x")


def main() raises:
    var ctx = DeviceContext()
    print("BatchNorm2D NCHW-vs-NHWC parity [fp32]")
    print("=" * 70)
    run_bn[64, 32, 48, 48, 5, 50](ctx, "EZv2 rep48")
    run_bn[64, 64, 24, 24, 5, 50](ctx, "EZv2 rep24")
    run_bn[256, 64, 6, 7, 5, 50](ctx, "C4 res")
    print("=" * 70)
    print("GO if: mismatches=0 AND NHWC-2D/realNCHW ~<=1.2x on the hot shapes.")
    print("Perf truth = NVIDIA; Apple is parity-only (2D skipped).")
