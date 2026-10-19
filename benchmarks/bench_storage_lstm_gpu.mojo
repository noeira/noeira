"""Group A (A1) microbench: storage LSTMCell forward GPU —
baseline (hand-rolled per-thread gate inner-product over IN_+H for all 4 gates,
no tiling) vs optimized (ix = x@W_ih, hx = h_prev@W_hh via max_matmul + an
elementwise gate/cell kernel). Self-contained A/B → real speedup on this GPU.

Compute-bound on the gate GEMMs → expect multiples on larger IN_/H.

Run (NVIDIA — perf sign-off):
    pixi run -e nvidia mojo run -I . benchmarks/bench_storage_lstm_gpu.mojo
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
def _sigmoid(x: Scalar[DT]) -> Scalar[DT]:
    if x >= 0:
        return Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-x))
    var e = exp(x)
    return e / (Scalar[DT](1.0) + e)


def _lstm_fwd_naive[
    BATCH: Int, IN_: Int, H: Int
](
    x: Tile[DT, BATCH, IN_],
    W_ih: Tile[DT, IN_, 4 * H],
    W_hh: Tile[DT, H, 4 * H],
    b: Tile[DT, 4 * H],
    h_prev: Tile[DT, BATCH, H],
    c_prev: Tile[DT, BATCH, H],
    h_t: Tile[DT, BATCH, H],
    c_t: Tile[DT, BATCH, H],
):
    var bi = Int(block_idx.x)
    if bi >= BATCH:
        return
    var j = Int(thread_idx.x)
    while j < H:
        var i_pre = Scalar[DT](0)
        var f_pre = Scalar[DT](0)
        var g_pre = Scalar[DT](0)
        var o_pre = Scalar[DT](0)
        for jj in range(IN_):
            var xv = rebind[Scalar[DT]](x[bi, jj])
            i_pre += xv * rebind[Scalar[DT]](W_ih[jj, j])
            f_pre += xv * rebind[Scalar[DT]](W_ih[jj, H + j])
            g_pre += xv * rebind[Scalar[DT]](W_ih[jj, 2 * H + j])
            o_pre += xv * rebind[Scalar[DT]](W_ih[jj, 3 * H + j])
        for jj in range(H):
            var hv = rebind[Scalar[DT]](h_prev[bi, jj])
            i_pre += hv * rebind[Scalar[DT]](W_hh[jj, j])
            f_pre += hv * rebind[Scalar[DT]](W_hh[jj, H + j])
            g_pre += hv * rebind[Scalar[DT]](W_hh[jj, 2 * H + j])
            o_pre += hv * rebind[Scalar[DT]](W_hh[jj, 3 * H + j])
        var i_val = _sigmoid(i_pre + rebind[Scalar[DT]](b[j]))
        var f_val = _sigmoid(f_pre + rebind[Scalar[DT]](b[H + j]))
        var g_val = tanh(g_pre + rebind[Scalar[DT]](b[2 * H + j]))
        var o_val = _sigmoid(o_pre + rebind[Scalar[DT]](b[3 * H + j]))
        var c_new = f_val * rebind[Scalar[DT]](c_prev[bi, j]) + i_val * g_val
        c_t[bi, j] = c_new
        h_t[bi, j] = o_val * tanh(c_new)
        j += TPB


def _lstm_gate[
    BATCH: Int, H: Int
](
    ix: Tile[DT, BATCH, 4 * H],
    hx: Tile[DT, BATCH, 4 * H],
    b: Tile[DT, 4 * H],
    c_prev: Tile[DT, BATCH, H],
    h_t: Tile[DT, BATCH, H],
    c_t: Tile[DT, BATCH, H],
):
    var gid = Int(global_idx.x)
    if gid >= BATCH * H:
        return
    var bi = gid // H
    var j = gid % H
    var i_v = _sigmoid(
        rebind[Scalar[DT]](ix[bi, j]) + rebind[Scalar[DT]](hx[bi, j])
        + rebind[Scalar[DT]](b[j])
    )
    var f_v = _sigmoid(
        rebind[Scalar[DT]](ix[bi, H + j]) + rebind[Scalar[DT]](hx[bi, H + j])
        + rebind[Scalar[DT]](b[H + j])
    )
    var g_v = tanh(
        rebind[Scalar[DT]](ix[bi, 2 * H + j])
        + rebind[Scalar[DT]](hx[bi, 2 * H + j])
        + rebind[Scalar[DT]](b[2 * H + j])
    )
    var o_v = _sigmoid(
        rebind[Scalar[DT]](ix[bi, 3 * H + j])
        + rebind[Scalar[DT]](hx[bi, 3 * H + j])
        + rebind[Scalar[DT]](b[3 * H + j])
    )
    var c_new = f_v * rebind[Scalar[DT]](c_prev[bi, j]) + i_v * g_v
    c_t[bi, j] = c_new
    h_t[bi, j] = o_v * tanh(c_new)


def _time[
    BATCH: Int, IN_: Int, H: Int, GEMM: Bool, WARMUP: Int, ITERS: Int
](ctx: DeviceContext, label: StaticString) raises:
    comptime FOURH = 4 * H
    var x = ctx.enqueue_create_buffer[DT](BATCH * IN_)
    var wih = ctx.enqueue_create_buffer[DT](IN_ * FOURH)
    var whh = ctx.enqueue_create_buffer[DT](H * FOURH)
    var bb = ctx.enqueue_create_buffer[DT](FOURH)
    var hp = ctx.enqueue_create_buffer[DT](BATCH * H)
    var cp = ctx.enqueue_create_buffer[DT](BATCH * H)
    var ht = ctx.enqueue_create_buffer[DT](BATCH * H)
    var ct = ctx.enqueue_create_buffer[DT](BATCH * H)
    var ix = ctx.enqueue_create_buffer[DT](BATCH * FOURH)
    var hx = ctx.enqueue_create_buffer[DT](BATCH * FOURH)
    _ = x.enqueue_fill(Scalar[DT](0.01))
    _ = wih.enqueue_fill(Scalar[DT](0.02))
    _ = whh.enqueue_fill(Scalar[DT](0.02))
    _ = bb.enqueue_fill(Scalar[DT](0.0))
    _ = hp.enqueue_fill(Scalar[DT](0.01))
    _ = cp.enqueue_fill(Scalar[DT](0.01))

    comptime lbi = RowMajor[BATCH, IN_]
    comptime lwih = RowMajor[IN_, FOURH]
    comptime lwhh = RowMajor[H, FOURH]
    comptime l4 = RowMajor[FOURH]
    comptime lbh = RowMajor[BATCH, H]
    comptime lb4 = RowMajor[BATCH, FOURH]
    var us = Float64(0)

    comptime if GEMM:
        var x_v = TileTensor(x, row_major[BATCH, IN_]())
        var wih_v = TileTensor(wih, row_major[IN_, FOURH]())
        var whh_v = TileTensor(whh, row_major[H, FOURH]())
        var hp_v = TileTensor(hp, row_major[BATCH, H]())
        var ix_v = TileTensor(ix, row_major[BATCH, FOURH]())
        var hx_v = TileTensor(hx, row_major[BATCH, FOURH]())
        var bl = tile[FOURH](bb)
        var cpl = tile[BATCH, H](cp)
        var htl = tile[BATCH, H](ht)
        var ctl = tile[BATCH, H](ct)
        var ixl = tile[BATCH, FOURH](ix)
        var hxl = tile[BATCH, FOURH](hx)
        comptime nblk = (BATCH * H + TPB - 1) // TPB

        comptime for _ in range(WARMUP):
            max_matmul[target="gpu"](ix_v, x_v, wih_v, ctx)
            max_matmul[target="gpu"](hx_v, hp_v, whh_v, ctx)
            ctx.enqueue_function[_lstm_gate[BATCH, H]](
                ixl, hxl, bl, cpl, htl, ctl, grid_dim=nblk, block_dim=TPB
            )
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            max_matmul[target="gpu"](ix_v, x_v, wih_v, ctx)
            max_matmul[target="gpu"](hx_v, hp_v, whh_v, ctx)
            ctx.enqueue_function[_lstm_gate[BATCH, H]](
                ixl, hxl, bl, cpl, htl, ctl, grid_dim=nblk, block_dim=TPB
            )
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    else:
        var xl = tile[BATCH, IN_](x)
        var wihl = tile[IN_, FOURH](wih)
        var whhl = tile[H, FOURH](whh)
        var bl = tile[FOURH](bb)
        var hpl = tile[BATCH, H](hp)
        var cpl = tile[BATCH, H](cp)
        var htl = tile[BATCH, H](ht)
        var ctl = tile[BATCH, H](ct)

        comptime for _ in range(WARMUP):
            ctx.enqueue_function[_lstm_fwd_naive[BATCH, IN_, H]](
                xl, wihl, whhl, bl, hpl, cpl, htl, ctl,
                grid_dim=BATCH, block_dim=TPB,
            )
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            ctx.enqueue_function[_lstm_fwd_naive[BATCH, IN_, H]](
                xl, wihl, whhl, bl, hpl, cpl, htl, ctl,
                grid_dim=BATCH, block_dim=TPB,
            )
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0

    var gflop = 8.0 * Float64(BATCH) * Float64(H) * Float64(IN_ + H) / 1e9
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
    print("LSTMCell forward GPU — naive gate inner-product vs max_matmul [fp32] (A1)")
    print("=" * 64)
    _ab[4096, 128, 128, 5, 50](ctx)
    _ab[4096, 256, 256, 5, 50](ctx)
    _ab[2048, 512, 512, 5, 50](ctx)
    _ab[4096, 512, 256, 5, 50](ctx)
    print("=" * 64)
    print("gemm/naive speedup = TFLOP/s ratio. Compute-bound: expect multiples.")
