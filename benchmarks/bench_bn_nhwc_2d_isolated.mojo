"""Minimal isolated test of the NHWC-2D BatchNorm reduction (one shape, no pool).

Purpose: the 2D kernel SIGILLs the Metal backend inside the big
`bench_bn_pool_nhwc_parity_gpu.mojo`. Hypothesis: that's a large-file comptime /
codegen-pressure crash, NOT the kernel itself. This file is deliberately tiny —
only the 2D stats/finalize/normalize kernels + one shape (EZv2 rep48) + a
host-side correctness check — to see if it compiles + runs on Apple when the
module is small. If it does, the 2D BN is portable (no NVIDIA gate needed).

Run (Apple):  pixi run -e apple  mojo run -I . benchmarks/bench_bn_nhwc_2d_isolated.mojo
Run (NVIDIA): pixi run -e nvidia mojo run -I . benchmarks/bench_bn_nhwc_2d_isolated.mojo
"""

from std.math import sqrt
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from noeira.core.tiles import Tile, tile

comptime DT = DType.float32
comptime TPB = 128
comptime NCHUNK = 256     # finalize-bound: fewer chunks = cheaper grid=C finalize
                          # (1024 > 512 > 256 in cost). Match the main BN bench tune.
comptime BN2D_BLK = 256        # = ROWS * C
comptime EPS = 1e-5

# one shape: EZv2 rep48
comptime N = 64
comptime C = 32
comptime H = 48
comptime W = 48
comptime SP = H * W
comptime R = N * SP            # rows = N*spatial
comptime RC = R * C
comptime ROWS = BN2D_BLK // C  # row-groups per block
comptime P = NCHUNK * ROWS     # partials per channel


# 2D partial: block_dim = BN2D_BLK = ROWS×C. Thread (rg,c) accumulates channel c
# over its strided row subset and writes its OWN partial — no shared mem.
def _bn_stats_nhwc_2d(
    input: Tile[DT, R * C],
    partial_sum: Tile[DT, NCHUNK * BN2D_BLK],
    partial_sumsq: Tile[DT, NCHUNK * BN2D_BLK],
):
    var chunk = Int(block_idx.x)
    var lane = Int(thread_idx.x)
    var c = lane % C
    var rg = lane // C
    comptime rpc = (R + NCHUNK - 1) // NCHUNK
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


# parallel finalize: grid=C, block reduces its P partials via block.sum.
def _bn_finalize_nhwc_parallel(
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
        var inv_n = Scalar[DT](1.0) / Scalar[DT](Float32(R))
        var mean = bs[0] * inv_n
        var var_ = bsq[0] * inv_n - mean * mean
        if var_ < Scalar[DT](0.0):
            var_ = Scalar[DT](0.0)
        mean_out[c] = mean
        inv_std_out[c] = Scalar[DT](1.0) / sqrt(var_ + Scalar[DT](EPS))


# flat normalize (coalesced both layouts): c = idx % C.
def _bn_norm_nhwc(
    input: Tile[DT, RC],
    output: Tile[DT, RC],
    mean_in: Tile[DT, C],
    inv_std_in: Tile[DT, C],
):
    var idx = Int(global_idx.x)
    if idx >= RC:
        return
    var c = idx % C
    var xh = (
        rebind[Scalar[DT]](input[idx]) - rebind[Scalar[DT]](mean_in[c])
    ) * rebind[Scalar[DT]](inv_std_in[c])
    output[idx] = xh


def main() raises:
    var ctx = DeviceContext()
    print("NHWC-2D BN isolated [rep48: N", N, "C", C, H, "x", W, "| R", R, "]")

    var x_buf = ctx.enqueue_create_buffer[DT](RC)
    var out_buf = ctx.enqueue_create_buffer[DT](RC)
    var ps = ctx.enqueue_create_buffer[DT](NCHUNK * BN2D_BLK)
    var psq = ctx.enqueue_create_buffer[DT](NCHUNK * BN2D_BLK)
    var mean_buf = ctx.enqueue_create_buffer[DT](C)
    var istd_buf = ctx.enqueue_create_buffer[DT](C)

    # NHWC fill: x[(b*SP+s)*C + c] distinct per (b,s,c)
    with x_buf.map_to_host() as hx:
        for r in range(R):
            for c in range(C):
                hx[r * C + c] = Scalar[DT](
                    Float64(((r * C + c) % 1009) + 1) * 0.03 - 15.0
                )

    var xt = tile[RC](x_buf)
    var xr = tile[R * C](x_buf)
    var ot = tile[RC](out_buf)
    var psl = tile[NCHUNK * BN2D_BLK](ps)
    var pql = tile[NCHUNK * BN2D_BLK](psq)
    var mt = tile[C](mean_buf)
    var it = tile[C](istd_buf)

    comptime nb_norm = (RC + TPB - 1) // TPB
    ctx.enqueue_function[_bn_stats_nhwc_2d](
        xr, psl, pql, grid_dim=NCHUNK, block_dim=BN2D_BLK)
    ctx.enqueue_function[_bn_finalize_nhwc_parallel](
        psl, pql, mt, it, grid_dim=C, block_dim=TPB)
    ctx.enqueue_function[_bn_norm_nhwc](
        xt, ot, mt, it, grid_dim=nb_norm, block_dim=TPB)
    ctx.synchronize()

    # host reference: per-channel mean / inv_std
    var bad = 0
    with x_buf.map_to_host() as hx:
        with mean_buf.map_to_host() as hm:
            with istd_buf.map_to_host() as hi:
                for c in range(C):
                    var s: Float64 = 0.0
                    var sq: Float64 = 0.0
                    for r in range(R):
                        var v = Float64(hx[r * C + c])
                        s += v
                        sq += v * v
                    var mean = s / Float64(R)
                    var var_ = sq / Float64(R) - mean * mean
                    if var_ < 0.0:
                        var_ = 0.0
                    var istd = 1.0 / (var_ + EPS) ** 0.5
                    if abs(Float64(hm[c]) - mean) > 1e-2:
                        bad += 1
                    if abs(Float64(hi[c]) - istd) > 1e-2:
                        bad += 1
    print("  verify (vs host): mismatches =", bad)

    # time the 2D pipeline
    comptime WARMUP = 5
    comptime ITERS = 50
    comptime for _ in range(WARMUP):
        ctx.enqueue_function[_bn_stats_nhwc_2d](
            xr, psl, pql, grid_dim=NCHUNK, block_dim=BN2D_BLK)
        ctx.enqueue_function[_bn_finalize_nhwc_parallel](
            psl, pql, mt, it, grid_dim=C, block_dim=TPB)
        ctx.enqueue_function[_bn_norm_nhwc](
            xt, ot, mt, it, grid_dim=nb_norm, block_dim=TPB)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    comptime for _ in range(ITERS):
        ctx.enqueue_function[_bn_stats_nhwc_2d](
            xr, psl, pql, grid_dim=NCHUNK, block_dim=BN2D_BLK)
        ctx.enqueue_function[_bn_finalize_nhwc_parallel](
            psl, pql, mt, it, grid_dim=C, block_dim=TPB)
        ctx.enqueue_function[_bn_norm_nhwc](
            xt, ot, mt, it, grid_dim=nb_norm, block_dim=TPB)
    ctx.synchronize()
    var us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    var gb = 2.0 * Float64(RC) * 4.0 / 1e9
    print("  NHWC-2D BN: ", us, "us  ", gb / (us / 1e6) / 1e3, "TB/s")
    print("DONE")
