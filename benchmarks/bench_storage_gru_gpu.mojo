"""Group A (A2) microbench: storage GRUCell forward GPU —
baseline (hand-rolled per-thread gate inner-product over IN_+H for all 3 gates)
vs optimized (ix = x@W_ih, hx = h_prev@W_hh via max_matmul + elementwise gate
kernel). Self-contained A/B. Compute-bound → expect multiples.

Run (NVIDIA): pixi run -e nvidia mojo run -I . benchmarks/bench_storage_gru_gpu.mojo
"""

from std.math import exp, tanh
from max.gpu import thread_idx, block_idx, global_idx
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from layout import TileTensor, row_major
from noeira.core.tiles import RowMajor, Tile, tile
from linalg.matmul import matmul as max_matmul

comptime DT = DType.float32
comptime TPB = 128


@always_inline
def _sig(x: Scalar[DT]) -> Scalar[DT]:
    if x >= 0:
        return Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-x))
    var e = exp(x)
    return e / (Scalar[DT](1.0) + e)


def _gru_naive[
    BATCH: Int, IN_: Int, H: Int
](
    x: Tile[DT, BATCH, IN_],
    W_ih: Tile[DT, IN_, 3 * H],
    W_hh: Tile[DT, H, 3 * H],
    b_ih: Tile[DT, 3 * H],
    b_hh: Tile[DT, 3 * H],
    h_prev: Tile[DT, BATCH, H],
    out_buf: Tile[DT, BATCH, H],
):
    var bi = Int(block_idx.x)
    if bi >= BATCH:
        return
    var j = Int(thread_idx.x)
    while j < H:
        var ir = rebind[Scalar[DT]](b_ih[j])
        var iz = rebind[Scalar[DT]](b_ih[H + j])
        var in_pre = rebind[Scalar[DT]](b_ih[2 * H + j])
        var hr = rebind[Scalar[DT]](b_hh[j])
        var hz = rebind[Scalar[DT]](b_hh[H + j])
        var hn = rebind[Scalar[DT]](b_hh[2 * H + j])
        for k in range(IN_):
            var xv = rebind[Scalar[DT]](x[bi, k])
            ir += xv * rebind[Scalar[DT]](W_ih[k, j])
            iz += xv * rebind[Scalar[DT]](W_ih[k, H + j])
            in_pre += xv * rebind[Scalar[DT]](W_ih[k, 2 * H + j])
        for k in range(H):
            var hv = rebind[Scalar[DT]](h_prev[bi, k])
            hr += hv * rebind[Scalar[DT]](W_hh[k, j])
            hz += hv * rebind[Scalar[DT]](W_hh[k, H + j])
            hn += hv * rebind[Scalar[DT]](W_hh[k, 2 * H + j])
        var rg = _sig(ir + hr)
        var zg = _sig(iz + hz)
        var ng = tanh(in_pre + rg * hn)
        out_buf[bi, j] = (
            (Scalar[DT](1.0) - zg) * ng + zg * rebind[Scalar[DT]](h_prev[bi, j])
        )
        j += TPB


def _gru_gate[
    BATCH: Int, H: Int
](
    ix: Tile[DT, BATCH, 3 * H],
    hx: Tile[DT, BATCH, 3 * H],
    b_ih: Tile[DT, 3 * H],
    b_hh: Tile[DT, 3 * H],
    h_prev: Tile[DT, BATCH, H],
    out_buf: Tile[DT, BATCH, H],
):
    var gid = Int(global_idx.x)
    if gid >= BATCH * H:
        return
    var bi = gid // H
    var j = gid % H
    var rg = _sig(
        rebind[Scalar[DT]](ix[bi, j]) + rebind[Scalar[DT]](b_ih[j])
        + rebind[Scalar[DT]](hx[bi, j]) + rebind[Scalar[DT]](b_hh[j])
    )
    var zg = _sig(
        rebind[Scalar[DT]](ix[bi, H + j]) + rebind[Scalar[DT]](b_ih[H + j])
        + rebind[Scalar[DT]](hx[bi, H + j]) + rebind[Scalar[DT]](b_hh[H + j])
    )
    var in_pre = rebind[Scalar[DT]](ix[bi, 2 * H + j]) + rebind[Scalar[DT]](
        b_ih[2 * H + j]
    )
    var hn = rebind[Scalar[DT]](hx[bi, 2 * H + j]) + rebind[Scalar[DT]](
        b_hh[2 * H + j]
    )
    var ng = tanh(in_pre + rg * hn)
    out_buf[bi, j] = (
        (Scalar[DT](1.0) - zg) * ng + zg * rebind[Scalar[DT]](h_prev[bi, j])
    )


def _time[
    BATCH: Int, IN_: Int, H: Int, GEMM: Bool, WARMUP: Int, ITERS: Int
](ctx: DeviceContext, label: StaticString) raises:
    comptime TH = 3 * H
    var x = ctx.enqueue_create_buffer[DT](BATCH * IN_)
    var wih = ctx.enqueue_create_buffer[DT](IN_ * TH)
    var whh = ctx.enqueue_create_buffer[DT](H * TH)
    var bih = ctx.enqueue_create_buffer[DT](TH)
    var bhh = ctx.enqueue_create_buffer[DT](TH)
    var hp = ctx.enqueue_create_buffer[DT](BATCH * H)
    var ob = ctx.enqueue_create_buffer[DT](BATCH * H)
    var ix = ctx.enqueue_create_buffer[DT](BATCH * TH)
    var hx = ctx.enqueue_create_buffer[DT](BATCH * TH)
    _ = x.enqueue_fill(Scalar[DT](0.01))
    _ = wih.enqueue_fill(Scalar[DT](0.02))
    _ = whh.enqueue_fill(Scalar[DT](0.02))
    _ = bih.enqueue_fill(Scalar[DT](0.0))
    _ = bhh.enqueue_fill(Scalar[DT](0.0))
    _ = hp.enqueue_fill(Scalar[DT](0.01))

    comptime lbi = RowMajor[BATCH, IN_]
    comptime lwih = RowMajor[IN_, TH]
    comptime lwhh = RowMajor[H, TH]
    comptime l3 = RowMajor[TH]
    comptime lbh = RowMajor[BATCH, H]
    comptime lb3 = RowMajor[BATCH, TH]
    var us = Float64(0)

    comptime if GEMM:
        var x_v = TileTensor(x, row_major[BATCH, IN_]())
        var wih_v = TileTensor(wih, row_major[IN_, TH]())
        var whh_v = TileTensor(whh, row_major[H, TH]())
        var hp_v = TileTensor(hp, row_major[BATCH, H]())
        var ix_v = TileTensor(ix, row_major[BATCH, TH]())
        var hx_v = TileTensor(hx, row_major[BATCH, TH]())
        var ixl = tile[BATCH, TH](ix)
        var hxl = tile[BATCH, TH](hx)
        var bihl = tile[TH](bih)
        var bhhl = tile[TH](bhh)
        var hpl = tile[BATCH, H](hp)
        var obl = tile[BATCH, H](ob)
        comptime nblk = (BATCH * H + TPB - 1) // TPB

        comptime for _ in range(WARMUP):
            max_matmul[target="gpu"](ix_v, x_v, wih_v, ctx)
            max_matmul[target="gpu"](hx_v, hp_v, whh_v, ctx)
            ctx.enqueue_function[_gru_gate[BATCH, H]](
                ixl, hxl, bihl, bhhl, hpl, obl, grid_dim=nblk, block_dim=TPB
            )
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            max_matmul[target="gpu"](ix_v, x_v, wih_v, ctx)
            max_matmul[target="gpu"](hx_v, hp_v, whh_v, ctx)
            ctx.enqueue_function[_gru_gate[BATCH, H]](
                ixl, hxl, bihl, bhhl, hpl, obl, grid_dim=nblk, block_dim=TPB
            )
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    else:
        var xl = tile[BATCH, IN_](x)
        var wihl = tile[IN_, TH](wih)
        var whhl = tile[H, TH](whh)
        var bihl = tile[TH](bih)
        var bhhl = tile[TH](bhh)
        var hpl = tile[BATCH, H](hp)
        var obl = tile[BATCH, H](ob)

        comptime for _ in range(WARMUP):
            ctx.enqueue_function[_gru_naive[BATCH, IN_, H]](
                xl, wihl, whhl, bihl, bhhl, hpl, obl,
                grid_dim=BATCH, block_dim=TPB,
            )
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            ctx.enqueue_function[_gru_naive[BATCH, IN_, H]](
                xl, wihl, whhl, bihl, bhhl, hpl, obl,
                grid_dim=BATCH, block_dim=TPB,
            )
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0

    var gflop = 6.0 * Float64(BATCH) * Float64(H) * Float64(IN_ + H) / 1e9
    print(
        "  ", label, " B=", BATCH, " IN=", IN_, " H=", H, " | ", us,
        "us/iter ", gflop / (us / 1e6) / 1e3, "TFLOP/s",
    )


def _ab[
    BATCH: Int, IN_: Int, H: Int, WARMUP: Int, ITERS: Int
](ctx: DeviceContext) raises:
    _time[BATCH, IN_, H, False, WARMUP, ITERS](ctx, "naive")
    _time[BATCH, IN_, H, True, WARMUP, ITERS](ctx, "gemm ")


def main() raises:
    var ctx = DeviceContext()
    print("GRUCell forward GPU — naive gate inner-product vs max_matmul [fp32] (A2)")
    print("=" * 64)
    _ab[4096, 128, 128, 5, 50](ctx)
    _ab[4096, 256, 256, 5, 50](ctx)
    _ab[2048, 512, 512, 5, 50](ctx)
    _ab[4096, 512, 256, 5, 50](ctx)
    print("=" * 64)
    print("gemm/naive speedup = TFLOP/s ratio. Compute-bound: expect multiples.")
