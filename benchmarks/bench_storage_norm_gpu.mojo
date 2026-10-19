"""D0′ microbench: storage LayerNorm forward GPU kernel —
baseline (scalar, 3× input re-read: sum / var / normalize) vs single-pass
(each thread reads its feature slice ONCE into registers, raw-moments mean/var,
normalize from registers → 1 input read), both in accum_type. Self-contained
A/B: both kernels run in the SAME process at the SAME shapes, so one run gives
the real speedup on this GPU.

Both variants keep the identical launch config (grid=BATCH, block=128) so the
ONLY difference measured is the read count — not the block shape.

The norm forward is memory-bound. Effective GB/s = 5·BATCH·DIM·4 / time, an
approximation (counts 3 input reads + 2 writes); identical denominator for both
variants so the ratio tracks the realized speedup. NOTE: the repeated input
reads in the baseline may be served from L2 (small hot row), so if single-pass
shows little gain, the kernel was L2/latency-bound, not DRAM-bound — that itself
is the finding.

Run (NVIDIA — required for the perf sign-off in
docs/STORAGE_NN_GPU_KERNEL_OPTIMIZATION.md):
    pixi run -e nvidia mojo run -I . benchmarks/bench_storage_norm_gpu.mojo
Apple (sanity only — Metal perf is not the sign-off):
    pixi run -e apple  mojo run -I . benchmarks/bench_storage_norm_gpu.mojo
"""

from std.math import sqrt
from max.gpu import thread_idx, block_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from std.utils.numerics import get_accum_type
from noeira.core.tiles import RowMajor, Tile, tile

comptime DT = DType.float32
comptime TPB = 128
comptime ACC = get_accum_type[DT]()
comptime LN_EPS: Scalar[DT] = 1e-5


# ───────────── baseline: scalar strided, DT accumulation ─────────────
def _ln_fwd_scalar[
    BATCH: Int, DIM: Int
](
    input: Tile[DT, BATCH, DIM],
    output: Tile[DT, BATCH, DIM],
    gamma: Tile[DT, DIM],
    beta: Tile[DT, DIM],
    cache_xhat: Tile[DT, BATCH, DIM],
    cache_inv_std: Tile[DT, BATCH],
):
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if b >= BATCH:
        return
    var inv_dim: Scalar[DT] = 1.0 / Float32(DIM)
    var my_sum: Scalar[DT] = 0.0
    var idx = t
    while idx < DIM:
        my_sum += rebind[Scalar[DT]](input[b, idx])
        idx += TPB
    var mean_val = (
        block.sum[block_size=TPB, broadcast=True](val=my_sum) * inv_dim
    )
    var my_var: Scalar[DT] = 0.0
    idx = t
    while idx < DIM:
        var diff = rebind[Scalar[DT]](input[b, idx]) - mean_val
        my_var += diff * diff
        idx += TPB
    var var_val = (
        block.sum[block_size=TPB, broadcast=True](val=my_var) * inv_dim
    )
    var inv_std: Scalar[DT] = 1.0 / sqrt(var_val + LN_EPS)
    if t == 0:
        cache_inv_std[b] = inv_std
    idx = t
    while idx < DIM:
        var x = rebind[Scalar[DT]](input[b, idx])
        var x_hat = (x - mean_val) * inv_std
        cache_xhat[b, idx] = x_hat
        output[b, idx] = rebind[Scalar[DT]](gamma[idx]) * x_hat + rebind[
            Scalar[DT]
        ](beta[idx])
        idx += TPB


# ──── optimized: single-pass register-cached (1 input read), accum_type ────
def _ln_fwd_single[
    BATCH: Int, DIM: Int
](
    input: Tile[DT, BATCH, DIM],
    output: Tile[DT, BATCH, DIM],
    gamma: Tile[DT, DIM],
    beta: Tile[DT, DIM],
    cache_xhat: Tile[DT, BATCH, DIM],
    cache_inv_std: Tile[DT, BATCH],
):
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if b >= BATCH:
        return
    comptime ELEMS = (DIM + TPB - 1) // TPB
    comptime REG_CACHE = ELEMS <= 8  # mirror LN_REG_CAP in the production kernel
    var inv_dim = Scalar[ACC](1.0) / Scalar[ACC](DIM)
    var my_sum = Scalar[ACC](0)
    var my_sumsq = Scalar[ACC](0)

    comptime if REG_CACHE:
        var slice = Array[Scalar[ACC], ELEMS](fill=Scalar[ACC](0))

        comptime for e in range(ELEMS):
            var col = t + e * TPB
            if col < DIM:
                var x = rebind[Scalar[DT]](input[b, col]).cast[ACC]()
                slice[e] = x
                my_sum += x
                my_sumsq += x * x
        var mean_val = (
            block.sum[block_size=TPB, broadcast=True](val=my_sum) * inv_dim
        )
        var ex2 = (
            block.sum[block_size=TPB, broadcast=True](val=my_sumsq) * inv_dim
        )
        var var_val = ex2 - mean_val * mean_val
        if var_val < Scalar[ACC](0):
            var_val = Scalar[ACC](0)
        var inv_std = Scalar[ACC](1.0) / sqrt(var_val + LN_EPS.cast[ACC]())
        if t == 0:
            cache_inv_std[b] = inv_std.cast[DT]()

        comptime for e in range(ELEMS):
            var col = t + e * TPB
            if col < DIM:
                var x_hat = (slice[e] - mean_val) * inv_std
                cache_xhat[b, col] = x_hat.cast[DT]()
                var g = rebind[Scalar[DT]](gamma[col]).cast[ACC]()
                var bt = rebind[Scalar[DT]](beta[col]).cast[ACC]()
                output[b, col] = (g * x_hat + bt).cast[DT]()
    else:
        var idx = t
        while idx < DIM:
            var x = rebind[Scalar[DT]](input[b, idx]).cast[ACC]()
            my_sum += x
            my_sumsq += x * x
            idx += TPB
        var mean_val = (
            block.sum[block_size=TPB, broadcast=True](val=my_sum) * inv_dim
        )
        var ex2 = (
            block.sum[block_size=TPB, broadcast=True](val=my_sumsq) * inv_dim
        )
        var var_val = ex2 - mean_val * mean_val
        if var_val < Scalar[ACC](0):
            var_val = Scalar[ACC](0)
        var inv_std = Scalar[ACC](1.0) / sqrt(var_val + LN_EPS.cast[ACC]())
        if t == 0:
            cache_inv_std[b] = inv_std.cast[DT]()
        idx = t
        while idx < DIM:
            var x = rebind[Scalar[DT]](input[b, idx]).cast[ACC]()
            var x_hat = (x - mean_val) * inv_std
            cache_xhat[b, idx] = x_hat.cast[DT]()
            var g = rebind[Scalar[DT]](gamma[idx]).cast[ACC]()
            var bt = rebind[Scalar[DT]](beta[idx]).cast[ACC]()
            output[b, idx] = (g * x_hat + bt).cast[DT]()
            idx += TPB


# ───────────── backward-dx: baseline (re-read) vs single (cached) ─────────────
def _ln_bwd_scalar[
    BATCH: Int, DIM: Int
](
    grad_output: Tile[DT, BATCH, DIM],
    gamma: Tile[DT, DIM],
    cache_xhat: Tile[DT, BATCH, DIM],
    cache_inv_std: Tile[DT, BATCH],
    grad_input: Tile[DT, BATCH, DIM],
):
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if b >= BATCH:
        return
    var inv_dim = Scalar[ACC](1.0) / Scalar[ACC](DIM)
    var inv_std = rebind[Scalar[DT]](cache_inv_std[b]).cast[ACC]()
    var my_g = Scalar[ACC](0)
    var my_g_xhat = Scalar[ACC](0)
    var idx = t
    while idx < DIM:
        var go = rebind[Scalar[DT]](grad_output[b, idx]).cast[ACC]()
        var gm = rebind[Scalar[DT]](gamma[idx]).cast[ACC]()
        var xh = rebind[Scalar[DT]](cache_xhat[b, idx]).cast[ACC]()
        var g = go * gm
        my_g += g
        my_g_xhat += g * xh
        idx += TPB
    var mean_g = block.sum[block_size=TPB, broadcast=True](val=my_g) * inv_dim
    var mean_g_xhat = (
        block.sum[block_size=TPB, broadcast=True](val=my_g_xhat) * inv_dim
    )
    idx = t
    while idx < DIM:
        var go = rebind[Scalar[DT]](grad_output[b, idx]).cast[ACC]()
        var gm = rebind[Scalar[DT]](gamma[idx]).cast[ACC]()
        var xh = rebind[Scalar[DT]](cache_xhat[b, idx]).cast[ACC]()
        var g = go * gm
        grad_input[b, idx] = (
            inv_std * (g - mean_g - xh * mean_g_xhat)
        ).cast[DT]()
        idx += TPB


def _ln_bwd_single[
    BATCH: Int, DIM: Int
](
    grad_output: Tile[DT, BATCH, DIM],
    gamma: Tile[DT, DIM],
    cache_xhat: Tile[DT, BATCH, DIM],
    cache_inv_std: Tile[DT, BATCH],
    grad_input: Tile[DT, BATCH, DIM],
):
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if b >= BATCH:
        return
    comptime ELEMS = (DIM + TPB - 1) // TPB
    comptime REG_CACHE = ELEMS <= 8
    var inv_dim = Scalar[ACC](1.0) / Scalar[ACC](DIM)
    var inv_std = rebind[Scalar[DT]](cache_inv_std[b]).cast[ACC]()
    var my_g = Scalar[ACC](0)
    var my_g_xhat = Scalar[ACC](0)

    comptime if REG_CACHE:
        var g_s = Array[Scalar[ACC], ELEMS](fill=Scalar[ACC](0))
        var xh_s = Array[Scalar[ACC], ELEMS](fill=Scalar[ACC](0))

        comptime for e in range(ELEMS):
            var col = t + e * TPB
            if col < DIM:
                var go = rebind[Scalar[DT]](grad_output[b, col]).cast[ACC]()
                var gm = rebind[Scalar[DT]](gamma[col]).cast[ACC]()
                var xh = rebind[Scalar[DT]](cache_xhat[b, col]).cast[ACC]()
                var g = go * gm
                g_s[e] = g
                xh_s[e] = xh
                my_g += g
                my_g_xhat += g * xh
        var mean_g = (
            block.sum[block_size=TPB, broadcast=True](val=my_g) * inv_dim
        )
        var mean_g_xhat = (
            block.sum[block_size=TPB, broadcast=True](val=my_g_xhat) * inv_dim
        )

        comptime for e in range(ELEMS):
            var col = t + e * TPB
            if col < DIM:
                grad_input[b, col] = (
                    inv_std * (g_s[e] - mean_g - xh_s[e] * mean_g_xhat)
                ).cast[DT]()
    else:
        var idx = t
        while idx < DIM:
            var go = rebind[Scalar[DT]](grad_output[b, idx]).cast[ACC]()
            var gm = rebind[Scalar[DT]](gamma[idx]).cast[ACC]()
            var xh = rebind[Scalar[DT]](cache_xhat[b, idx]).cast[ACC]()
            var g = go * gm
            my_g += g
            my_g_xhat += g * xh
            idx += TPB
        var mean_g = (
            block.sum[block_size=TPB, broadcast=True](val=my_g) * inv_dim
        )
        var mean_g_xhat = (
            block.sum[block_size=TPB, broadcast=True](val=my_g_xhat) * inv_dim
        )
        idx = t
        while idx < DIM:
            var go = rebind[Scalar[DT]](grad_output[b, idx]).cast[ACC]()
            var gm = rebind[Scalar[DT]](gamma[idx]).cast[ACC]()
            var xh = rebind[Scalar[DT]](cache_xhat[b, idx]).cast[ACC]()
            var g = go * gm
            grad_input[b, idx] = (
                inv_std * (g - mean_g - xh * mean_g_xhat)
            ).cast[DT]()
            idx += TPB


def _time[
    BATCH: Int, DIM: Int, VECTORIZED: Bool, WARMUP: Int, ITERS: Int
](ctx: DeviceContext, label: StaticString) raises:
    var inp = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    var out = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    var gam = ctx.enqueue_create_buffer[DT](DIM)
    var bet = ctx.enqueue_create_buffer[DT](DIM)
    var xh = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    var iv = ctx.enqueue_create_buffer[DT](BATCH)
    _ = inp.enqueue_fill(Scalar[DT](0.013))
    _ = gam.enqueue_fill(Scalar[DT](1.0))
    _ = bet.enqueue_fill(Scalar[DT](0.0))

    comptime l2d = RowMajor[BATCH, DIM]
    comptime ld = RowMajor[DIM]
    comptime lb = RowMajor[BATCH]
    var it = tile[BATCH, DIM](inp)
    var ot = tile[BATCH, DIM](out)
    var gt = tile[DIM](gam)
    var btt = tile[DIM](bet)
    var xt = tile[BATCH, DIM](xh)
    var ivt = tile[BATCH](iv)

    comptime kern = _ln_fwd_single[
        BATCH, DIM
    ] if VECTORIZED else _ln_fwd_scalar[BATCH, DIM]
    comptime for _ in range(WARMUP):
        ctx.enqueue_function[kern](
            it, ot, gt, btt, xt, ivt, grid_dim=BATCH, block_dim=TPB
        )
    ctx.synchronize()
    var t0 = perf_counter_ns()
    comptime for _ in range(ITERS):
        ctx.enqueue_function[kern](
            it, ot, gt, btt, xt, ivt, grid_dim=BATCH, block_dim=TPB
        )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var us = Float64(t1 - t0) / Float64(ITERS) / 1000.0
    var gb = 5.0 * Float64(BATCH) * Float64(DIM) * 4.0 / 1e9
    var gbps = gb / (us / 1e6)
    print(
        "  ",
        label,
        " B=",
        BATCH,
        " D=",
        DIM,
        " | ",
        us,
        "us/iter ",
        gbps,
        "GB/s",
    )


def _ab[
    BATCH: Int, DIM: Int, WARMUP: Int, ITERS: Int
](ctx: DeviceContext) raises:
    _time[BATCH, DIM, False, WARMUP, ITERS](ctx, "scalar")
    _time[BATCH, DIM, True, WARMUP, ITERS](ctx, "single")


def _time_bwd[
    BATCH: Int, DIM: Int, CACHED: Bool, WARMUP: Int, ITERS: Int
](ctx: DeviceContext, label: StaticString) raises:
    var go = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    var gam = ctx.enqueue_create_buffer[DT](DIM)
    var xh = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    var iv = ctx.enqueue_create_buffer[DT](BATCH)
    var gi = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    _ = go.enqueue_fill(Scalar[DT](0.01))
    _ = gam.enqueue_fill(Scalar[DT](1.0))
    _ = xh.enqueue_fill(Scalar[DT](0.5))
    _ = iv.enqueue_fill(Scalar[DT](1.0))

    comptime l2d = RowMajor[BATCH, DIM]
    comptime ld = RowMajor[DIM]
    comptime lb = RowMajor[BATCH]
    var got = tile[BATCH, DIM](go)
    var gt = tile[DIM](gam)
    var xt = tile[BATCH, DIM](xh)
    var ivt = tile[BATCH](iv)
    var git = tile[BATCH, DIM](gi)

    comptime kern = _ln_bwd_single[
        BATCH, DIM
    ] if CACHED else _ln_bwd_scalar[BATCH, DIM]
    comptime for _ in range(WARMUP):
        ctx.enqueue_function[kern](
            got, gt, xt, ivt, git, grid_dim=BATCH, block_dim=TPB
        )
    ctx.synchronize()
    var t0 = perf_counter_ns()
    comptime for _ in range(ITERS):
        ctx.enqueue_function[kern](
            got, gt, xt, ivt, git, grid_dim=BATCH, block_dim=TPB
        )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var us = Float64(t1 - t0) / Float64(ITERS) / 1000.0
    var gb = 4.0 * Float64(BATCH) * Float64(DIM) * 4.0 / 1e9
    var gbps = gb / (us / 1e6)
    print(
        "  ", label, " B=", BATCH, " D=", DIM, " | ", us, "us/iter ", gbps,
        "GB/s",
    )


def _ab_bwd[
    BATCH: Int, DIM: Int, WARMUP: Int, ITERS: Int
](ctx: DeviceContext) raises:
    _time_bwd[BATCH, DIM, False, WARMUP, ITERS](ctx, "scalar")
    _time_bwd[BATCH, DIM, True, WARMUP, ITERS](ctx, "single")


# ───────────── MinMaxNorm forward: block.min/max reduction (D1) ─────────────
def _mmn_fwd_scalar[
    BATCH: Int, DIM: Int
](
    input: Tile[DT, BATCH, DIM],
    output: Tile[DT, BATCH, DIM],
    cache: Tile[DT, BATCH, DIM],
):
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if b >= BATCH:
        return
    var my_min = Scalar[DT](1e30)
    var my_max = Scalar[DT](-1e30)
    var idx = t
    while idx < DIM:
        var v = rebind[Scalar[DT]](input[b, idx])
        if v < my_min:
            my_min = v
        if v > my_max:
            my_max = v
        idx += TPB
    var min_val = block.min[block_size=TPB, broadcast=True](val=my_min)
    var max_val = block.max[block_size=TPB, broadcast=True](val=my_max)
    var s = max_val - min_val
    if s < Scalar[DT](1e-5):
        s = Scalar[DT](1e-5)
    var inv_s = Scalar[DT](1.0) / s
    idx = t
    while idx < DIM:
        var x = rebind[Scalar[DT]](input[b, idx])
        cache[b, idx] = x
        output[b, idx] = (x - min_val) * inv_s
        idx += TPB


def _mmn_fwd_single[
    BATCH: Int, DIM: Int
](
    input: Tile[DT, BATCH, DIM],
    output: Tile[DT, BATCH, DIM],
    cache: Tile[DT, BATCH, DIM],
):
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if b >= BATCH:
        return
    comptime ELEMS = (DIM + TPB - 1) // TPB
    var my_min = Scalar[DT](1e30)
    var my_max = Scalar[DT](-1e30)
    var slice = Array[Scalar[DT], ELEMS](fill=Scalar[DT](0))

    comptime for e in range(ELEMS):
        var col = t + e * TPB
        if col < DIM:
            var v = rebind[Scalar[DT]](input[b, col])
            slice[e] = v
            if v < my_min:
                my_min = v
            if v > my_max:
                my_max = v
    var min_val = block.min[block_size=TPB, broadcast=True](val=my_min)
    var max_val = block.max[block_size=TPB, broadcast=True](val=my_max)
    var s = max_val - min_val
    if s < Scalar[DT](1e-5):
        s = Scalar[DT](1e-5)
    var inv_s = Scalar[DT](1.0) / s

    comptime for e in range(ELEMS):
        var col = t + e * TPB
        if col < DIM:
            cache[b, col] = slice[e]
            output[b, col] = (slice[e] - min_val) * inv_s


def _time_mmn[
    BATCH: Int, DIM: Int, CACHED: Bool, WARMUP: Int, ITERS: Int
](ctx: DeviceContext, label: StaticString) raises:
    var inp = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    var out = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    var ca = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    _ = inp.enqueue_fill(Scalar[DT](0.013))

    comptime l2d = RowMajor[BATCH, DIM]
    var it = tile[BATCH, DIM](inp)
    var ot = tile[BATCH, DIM](out)
    var ct = tile[BATCH, DIM](ca)

    comptime kern = _mmn_fwd_single[
        BATCH, DIM
    ] if CACHED else _mmn_fwd_scalar[BATCH, DIM]
    comptime for _ in range(WARMUP):
        ctx.enqueue_function[kern](
            it, ot, ct, grid_dim=BATCH, block_dim=TPB
        )
    ctx.synchronize()
    var t0 = perf_counter_ns()
    comptime for _ in range(ITERS):
        ctx.enqueue_function[kern](
            it, ot, ct, grid_dim=BATCH, block_dim=TPB
        )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var us = Float64(t1 - t0) / Float64(ITERS) / 1000.0
    var gb = 3.0 * Float64(BATCH) * Float64(DIM) * 4.0 / 1e9
    var gbps = gb / (us / 1e6)
    print(
        "  ", label, " B=", BATCH, " D=", DIM, " | ", us, "us/iter ", gbps,
        "GB/s",
    )


def _ab_mmn[
    BATCH: Int, DIM: Int, WARMUP: Int, ITERS: Int
](ctx: DeviceContext) raises:
    _time_mmn[BATCH, DIM, False, WARMUP, ITERS](ctx, "scalar")
    _time_mmn[BATCH, DIM, True, WARMUP, ITERS](ctx, "single")


def main() raises:
    var ctx = DeviceContext()
    print("LayerNorm forward GPU — baseline(scalar 3-read) vs single(1-read) [fp32]")
    print("=" * 66)
    _ab[4096, 256, 10, 100](ctx)
    _ab[4096, 512, 10, 100](ctx)
    _ab[4096, 1024, 10, 100](ctx)
    _ab[1024, 4096, 10, 100](ctx)
    _ab[16384, 256, 10, 100](ctx)
    print("-" * 66)
    print("LayerNorm backward-dx GPU — baseline(re-read) vs single(cached) [fp32]")
    print("-" * 66)
    _ab_bwd[4096, 256, 10, 100](ctx)
    _ab_bwd[4096, 512, 10, 100](ctx)
    _ab_bwd[4096, 1024, 10, 100](ctx)
    _ab_bwd[1024, 4096, 10, 100](ctx)
    _ab_bwd[16384, 256, 10, 100](ctx)
    print("-" * 66)
    print("MinMaxNorm forward GPU — baseline(2-read) vs single(1-read) [fp32] (D1)")
    print("-" * 66)
    _ab_mmn[4096, 256, 10, 100](ctx)
    _ab_mmn[4096, 512, 10, 100](ctx)
    _ab_mmn[4096, 1024, 10, 100](ctx)
    _ab_mmn[16384, 256, 10, 100](ctx)
    print("=" * 66)
    print("single/scalar speedup = µs ratio. Forward + backward-dx both cut the")
    print("input re-read; if single≈scalar the kernel was L2/latency-bound.")
