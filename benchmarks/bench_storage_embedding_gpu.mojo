"""Group A (A4) microbench: storage Embedding forward GPU —
baseline (hand-rolled naive one-thread-per-output, O(VOCAB) serial inner loop)
vs optimized (max_matmul tiled GEMM). out[B,ED] = input[B,VOCAB] @ weight[VOCAB,ED].
Self-contained A/B in one process → real speedup on this GPU.

This is compute-bound (unlike the bandwidth-bound norms), so the upside is the
GEMM's tiling/reuse, not a 1.2× read-cut — expect multiples on large VOCAB.

Run (NVIDIA — perf sign-off):
    pixi run -e nvidia mojo run -I . benchmarks/bench_storage_embedding_gpu.mojo
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from layout import TileTensor, row_major
from noeira.core.tiles import RowMajor, Tile, tile
from linalg.matmul import matmul as max_matmul

comptime DT = DType.float32
comptime TPB = 128


def _emb_fwd_naive[
    B: Int, VOCAB: Int, ED: Int
](
    input: Tile[DT, B, VOCAB],
    weight: Tile[DT, VOCAB, ED],
    output: Tile[DT, B, ED],
):
    var gid = Int(global_idx.x)
    if gid >= B * ED:
        return
    var b = gid // ED
    var j = gid % ED
    var acc: Scalar[DT] = 0.0
    for v in range(VOCAB):
        acc += rebind[Scalar[DT]](input[b, v]) * rebind[Scalar[DT]](
            weight[v, j]
        )
    output[b, j] = acc


def _time[
    B: Int, VOCAB: Int, ED: Int, GEMM: Bool, WARMUP: Int, ITERS: Int
](ctx: DeviceContext, label: StaticString) raises:
    var inp = ctx.enqueue_create_buffer[DT](B * VOCAB)
    var w = ctx.enqueue_create_buffer[DT](VOCAB * ED)
    var out = ctx.enqueue_create_buffer[DT](B * ED)
    _ = inp.enqueue_fill(Scalar[DT](0.01))
    _ = w.enqueue_fill(Scalar[DT](0.02))

    comptime lbv = RowMajor[B, VOCAB]
    comptime lw = RowMajor[VOCAB, ED]
    comptime lbe = RowMajor[B, ED]

    comptime if GEMM:
        var in_v = TileTensor(inp, row_major[B, VOCAB]())
        var w_v = TileTensor(w, row_major[VOCAB, ED]())
        var out_v = TileTensor(out, row_major[B, ED]())
        comptime for _ in range(WARMUP):
            max_matmul[target="gpu"](out_v, in_v, w_v, ctx)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            max_matmul[target="gpu"](out_v, in_v, w_v, ctx)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var us = Float64(t1 - t0) / Float64(ITERS) / 1000.0
        var gflop = 2.0 * Float64(B) * Float64(VOCAB) * Float64(ED) / 1e9
        print(
            "  ", label, " B=", B, " V=", VOCAB, " ED=", ED, " | ", us,
            "us/iter ", gflop / (us / 1e6) / 1e3, "TFLOP/s",
        )
    else:
        var it = tile[B, VOCAB](inp)
        var wt = tile[VOCAB, ED](w)
        var ot = tile[B, ED](out)
        comptime nblk = (B * ED + TPB - 1) // TPB
        comptime for _ in range(WARMUP):
            ctx.enqueue_function[_emb_fwd_naive[B, VOCAB, ED]](
                it, wt, ot, grid_dim=nblk, block_dim=TPB
            )
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            ctx.enqueue_function[_emb_fwd_naive[B, VOCAB, ED]](
                it, wt, ot, grid_dim=nblk, block_dim=TPB
            )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var us = Float64(t1 - t0) / Float64(ITERS) / 1000.0
        var gflop = 2.0 * Float64(B) * Float64(VOCAB) * Float64(ED) / 1e9
        print(
            "  ", label, " B=", B, " V=", VOCAB, " ED=", ED, " | ", us,
            "us/iter ", gflop / (us / 1e6) / 1e3, "TFLOP/s",
        )


def _ab[
    B: Int, VOCAB: Int, ED: Int, WARMUP: Int, ITERS: Int
](ctx: DeviceContext) raises:
    _time[B, VOCAB, ED, False, WARMUP, ITERS](ctx, "naive ")
    _time[B, VOCAB, ED, True, WARMUP, ITERS](ctx, "gemm  ")


def main() raises:
    var ctx = DeviceContext()
    print("Embedding forward GPU — naive vs max_matmul [fp32] (A4)")
    print("=" * 60)
    _ab[4096, 256, 256, 5, 50](ctx)
    _ab[4096, 512, 256, 5, 50](ctx)
    _ab[4096, 1024, 256, 5, 50](ctx)
    _ab[16384, 256, 128, 5, 50](ctx)
    print("=" * 60)
    print("gemm/naive speedup = TFLOP/s ratio. Compute-bound: expect multiples.")
