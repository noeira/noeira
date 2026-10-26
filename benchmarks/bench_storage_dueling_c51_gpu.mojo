"""Group C (C3) microbench: storage DuelingHeadC51 combine GPU —
naive (one thread per (b,atom), reads the NA-wide advantage slab TWICE: once
to sum for the mean, once to write each Q) vs register-cached (reads the NA
advantages once into an Array, reuses for both the sum and the writes).
Self-contained A/B in one process.

The audit flagged "double serial loop, no parallelism over actions", but the
grid already runs BATCH·N_ATOMS threads (~26k at BATCH=512, N_ATOMS=51) and
NA is tiny (Atari 4–18) — so this is expected to be an A3/B3-style no-op
(memory-bound, already parallel). The only lever is halving the advantage
reads; this A/B measures whether that matters.

  Q[b,a,k] = V[b,k] + A[b,a,k] − (1/NA)·Σ_a A[b,a,k]

Shapes: C51/Rainbow N_ATOMS=51; NA = action count (Pong 6, full Atari 18);
BATCH = replay minibatch (32–512).

Run (NVIDIA): pixi run -e nvidia mojo run -I . benchmarks/bench_storage_dueling_c51_gpu.mojo
Run (Apple):  pixi run -e apple  mojo run -I . benchmarks/bench_storage_dueling_c51_gpu.mojo
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from noeira.core.tiles import Tile, tile

comptime DT = DType.float32
comptime TPB = 128
comptime NA_CAP = 32   # register-cache Array bound (Atari NA ≤ 18)


def _combine_naive[
    BATCH: Int, NA: Int, N_ATOMS: Int
](
    raw_in: Tile[DT, BATCH, (1 + NA) * N_ATOMS],
    q_out: Tile[DT, BATCH, NA * N_ATOMS],
):
    var lin = Int(global_idx.x)
    var total = BATCH * N_ATOMS
    if lin < total:
        var b = lin // N_ATOMS
        var k = lin % N_ATOMS
        var v_k = rebind[Scalar[DT]](raw_in[b, k])
        var sum_a: Scalar[DT] = 0.0
        for a in range(NA):
            sum_a += rebind[Scalar[DT]](raw_in[b, N_ATOMS + a * N_ATOMS + k])
        var mean_a = sum_a * (Scalar[DT](1.0) / Scalar[DT](NA))
        for a in range(NA):
            var adv = rebind[Scalar[DT]](raw_in[b, N_ATOMS + a * N_ATOMS + k])
            q_out[b, a * N_ATOMS + k] = v_k + (adv - mean_a)


def _combine_regcache[
    BATCH: Int, NA: Int, N_ATOMS: Int
](
    raw_in: Tile[DT, BATCH, (1 + NA) * N_ATOMS],
    q_out: Tile[DT, BATCH, NA * N_ATOMS],
):
    var lin = Int(global_idx.x)
    var total = BATCH * N_ATOMS
    if lin < total:
        var b = lin // N_ATOMS
        var k = lin % N_ATOMS
        var v_k = rebind[Scalar[DT]](raw_in[b, k])
        var adv = Array[Scalar[DT], NA](uninitialized=True)
        var sum_a: Scalar[DT] = 0.0
        for a in range(NA):
            var x = rebind[Scalar[DT]](raw_in[b, N_ATOMS + a * N_ATOMS + k])
            adv[a] = x
            sum_a += x
        var mean_a = sum_a * (Scalar[DT](1.0) / Scalar[DT](NA))
        for a in range(NA):
            q_out[b, a * N_ATOMS + k] = v_k + (adv[a] - mean_a)


def _time[
    BATCH: Int, NA: Int, N_ATOMS: Int, REG: Bool, WARMUP: Int, ITERS: Int
](ctx: DeviceContext, label: StaticString) raises:
    comptime IN_SIZE = (1 + NA) * N_ATOMS
    comptime OUT_SIZE = NA * N_ATOMS
    var ri = ctx.enqueue_create_buffer[DT](BATCH * IN_SIZE)
    var qo = ctx.enqueue_create_buffer[DT](BATCH * OUT_SIZE)
    _ = ri.enqueue_fill(Scalar[DT](0.01)); _ = qo.enqueue_fill(Scalar[DT](0.0))
    var ril = tile[BATCH, IN_SIZE](ri)
    var qol = tile[BATCH, OUT_SIZE](qo)
    comptime nb = (BATCH * N_ATOMS + TPB - 1) // TPB
    var us = Float64(0)
    comptime if REG:
        comptime for _ in range(WARMUP):
            ctx.enqueue_function[_combine_regcache[BATCH, NA, N_ATOMS]](
                ril, qol, grid_dim=nb, block_dim=TPB)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            ctx.enqueue_function[_combine_regcache[BATCH, NA, N_ATOMS]](
                ril, qol, grid_dim=nb, block_dim=TPB)
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    else:
        comptime for _ in range(WARMUP):
            ctx.enqueue_function[_combine_naive[BATCH, NA, N_ATOMS]](
                ril, qol, grid_dim=nb, block_dim=TPB)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            ctx.enqueue_function[_combine_naive[BATCH, NA, N_ATOMS]](
                ril, qol, grid_dim=nb, block_dim=TPB)
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    # bytes: read (1+NA)·N_ATOMS, write NA·N_ATOMS per sample, 4 bytes
    var gb = Float64(BATCH) * Float64(IN_SIZE + OUT_SIZE) * 4.0 / 1e9
    print("  ", label, " B=", BATCH, " NA=", NA, " ATOMS=", N_ATOMS, " | ",
          us, "us/iter ", gb / (us / 1e6) / 1e3, "TB/s")


def _ab[
    BATCH: Int, NA: Int, N_ATOMS: Int, WARMUP: Int, ITERS: Int
](ctx: DeviceContext) raises:
    _time[BATCH, NA, N_ATOMS, False, WARMUP, ITERS](ctx, "naive   ")
    _time[BATCH, NA, N_ATOMS, True, WARMUP, ITERS](ctx, "regcache")


def main() raises:
    var ctx = DeviceContext()
    print("DuelingHeadC51 combine GPU — naive (2 reads) vs regcache (1 read) (C3)")
    print("=" * 66)
    _ab[512, 6, 51, 5, 200](ctx)
    _ab[512, 18, 51, 5, 200](ctx)
    _ab[32, 4, 51, 5, 200](ctx)
    _ab[256, 18, 51, 5, 200](ctx)
    print("=" * 66)
    print("regcache/naive ~1.0 = memory-bound + already parallel (close as no-op).")
