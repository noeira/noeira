"""Group A (A3) microbench: storage BlockLinear grad_weight GPU —
baseline (one thread per weight element, serial reduction over BATCH) vs
optimized (one block per weight element, cooperative block.sum over BATCH).
This is the kernel the audit flagged as O(W_SIZE·BATCH) serial. Self-contained
A/B. DreamerV3 block-diagonal shapes (IN=DETER, OUT=3·DETER, BLOCKS=8).

Run (NVIDIA): pixi run -e nvidia mojo run -I . benchmarks/bench_storage_block_linear_gpu.mojo
"""

from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from noeira.core.tiles import Tile, tile

comptime DT = DType.float32
comptime TPB = 128


def _dw_serial[
    BATCH: Int, IN: Int, OUT: Int, BLK: Int
](
    x: Tile[DT, BATCH * IN],
    go: Tile[DT, BATCH * OUT],
    grad_w: Tile[DT, BLK * (IN // BLK) * (OUT // BLK)],
):
    comptime IPB = IN // BLK
    comptime OPB = OUT // BLK
    var idx = Int(global_idx.x)
    if idx >= BLK * IPB * OPB:
        return
    var k = idx // (IPB * OPB)
    var rem = idx % (IPB * OPB)
    var in_col = k * IPB + rem // OPB
    var out_col = k * OPB + rem % OPB
    var acc: Scalar[DT] = 0.0
    for b in range(BATCH):
        acc += rebind[Scalar[DT]](x[b * IN + in_col]) * rebind[Scalar[DT]](
            go[b * OUT + out_col]
        )
    grad_w[idx] = rebind[Scalar[DT]](grad_w[idx]) + acc


def _dw_blockreduce[
    BATCH: Int, IN: Int, OUT: Int, BLK: Int
](
    x: Tile[DT, BATCH * IN],
    go: Tile[DT, BATCH * OUT],
    grad_w: Tile[DT, BLK * (IN // BLK) * (OUT // BLK)],
):
    comptime IPB = IN // BLK
    comptime OPB = OUT // BLK
    var idx = Int(block_idx.x)
    if idx >= BLK * IPB * OPB:
        return
    var k = idx // (IPB * OPB)
    var rem = idx % (IPB * OPB)
    var in_col = k * IPB + rem // OPB
    var out_col = k * OPB + rem % OPB
    var my: Scalar[DT] = 0.0
    var b = Int(thread_idx.x)
    while b < BATCH:
        my += rebind[Scalar[DT]](x[b * IN + in_col]) * rebind[Scalar[DT]](
            go[b * OUT + out_col]
        )
        b += TPB
    var total = block.sum[block_size=TPB, broadcast=False](val=my)
    if Int(thread_idx.x) == 0:
        grad_w[idx] = rebind[Scalar[DT]](grad_w[idx]) + total[0]


def _time[
    BATCH: Int, IN: Int, OUT: Int, BLK: Int, REDUCE: Bool, WARMUP: Int, ITERS: Int
](ctx: DeviceContext, label: StaticString) raises:
    comptime WS = BLK * (IN // BLK) * (OUT // BLK)
    var x = ctx.enqueue_create_buffer[DT](BATCH * IN)
    var go = ctx.enqueue_create_buffer[DT](BATCH * OUT)
    var gw = ctx.enqueue_create_buffer[DT](WS)
    _ = x.enqueue_fill(Scalar[DT](0.01))
    _ = go.enqueue_fill(Scalar[DT](0.01))
    _ = gw.enqueue_fill(Scalar[DT](0.0))
    var xl = tile[BATCH * IN](x)
    var gol = tile[BATCH * OUT](go)
    var gwl = tile[WS](gw)
    var us = Float64(0)

    comptime if REDUCE:
        comptime for _ in range(WARMUP):
            ctx.enqueue_function[_dw_blockreduce[BATCH, IN, OUT, BLK]](
                xl, gol, gwl, grid_dim=WS, block_dim=TPB
            )
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            ctx.enqueue_function[_dw_blockreduce[BATCH, IN, OUT, BLK]](
                xl, gol, gwl, grid_dim=WS, block_dim=TPB
            )
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    else:
        comptime nb = (WS + TPB - 1) // TPB
        comptime for _ in range(WARMUP):
            ctx.enqueue_function[_dw_serial[BATCH, IN, OUT, BLK]](
                xl, gol, gwl, grid_dim=nb, block_dim=TPB
            )
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            ctx.enqueue_function[_dw_serial[BATCH, IN, OUT, BLK]](
                xl, gol, gwl, grid_dim=nb, block_dim=TPB
            )
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0

    var gflop = 2.0 * Float64(WS) * Float64(BATCH) / 1e9
    print(
        "  ", label, " B=", BATCH, " IN=", IN, " OUT=", OUT, " BLK=", BLK,
        " | ", us, "us/iter ", gflop / (us / 1e6) / 1e3, "TFLOP/s",
    )


def _ab[
    BATCH: Int, IN: Int, OUT: Int, BLK: Int, WARMUP: Int, ITERS: Int
](ctx: DeviceContext) raises:
    _time[BATCH, IN, OUT, BLK, False, WARMUP, ITERS](ctx, "serial")
    _time[BATCH, IN, OUT, BLK, True, WARMUP, ITERS](ctx, "reduce")


def main() raises:
    var ctx = DeviceContext()
    print("BlockLinear grad_weight GPU — serial-over-BATCH vs block.sum [fp32] (A3)")
    print("=" * 66)
    _ab[1024, 512, 1536, 8, 5, 50](ctx)
    _ab[4096, 512, 1536, 8, 5, 50](ctx)
    _ab[1024, 1024, 3072, 8, 5, 50](ctx)
    _ab[4096, 1024, 3072, 8, 5, 20](ctx)
    print("=" * 66)
    print("reduce/serial speedup = the serial BATCH reduction was the bottleneck.")
