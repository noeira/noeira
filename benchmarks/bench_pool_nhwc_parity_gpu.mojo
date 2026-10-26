"""MaxPool2D — NCHW vs NHWC parity (perf + correctness). [Pool-only split]

Split out of the former bench_bn_pool_nhwc_parity to cut compile time. BN parity
lives in bench_bn_nhwc_parity_gpu.mojo.

MaxPool is thread-per-output-element in both layouts; map consecutive threads to
the contiguous output dim (W in NCHW, C in NHWC) → coalesced either way. Expect
~parity. The bench answers GO/NO-GO (mismatch=0 AND ~≤1.2x).

Run (NVIDIA = perf truth):
    pixi run -e nvidia mojo run -I . benchmarks/bench_pool_nhwc_parity_gpu.mojo
Run (Apple = parity only):
    pixi run -e apple  mojo run -I . benchmarks/bench_pool_nhwc_parity_gpu.mojo
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from std.time import perf_counter_ns
from noeira.core.tiles import Tile, tile

comptime DT = DType.float32
comptime TPB = 128


# NCHW: out_pos = c*OSP + oh*OW + ow → consecutive threads = consecutive ow.
def _maxpool_nchw[
    N: Int, C: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, OUT_FLAT: Int,
](
    input: Tile[DT, N, IN_FLAT],
    output: Tile[DT, N, OUT_FLAT],
):
    var idx = Int(global_idx.x)
    if idx >= N * OUT_FLAT:
        return
    var b = idx // OUT_FLAT
    var out_pos = idx % OUT_FLAT
    var osp = OH * OW
    var c = out_pos // osp
    var rem = out_pos % osp
    var oh = rem // OW
    var ow = rem % OW
    var c_off = c * H * W
    var best: Scalar[DT] = -3.0e38
    for kh in range(K):
        var ih = oh * S + kh - P
        if ih < 0 or ih >= H:
            continue
        for kw in range(K):
            var iw = ow * S + kw - P
            if iw < 0 or iw >= W:
                continue
            var v = rebind[Scalar[DT]](input[b, c_off + ih * W + iw])
            if v > best:
                best = v
    output[b, out_pos] = best


# NHWC: out_pos = (oh*OW+ow)*C + c → consecutive threads = consecutive c.
def _maxpool_nhwc[
    N: Int, C: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, OUT_FLAT: Int,
](
    input: Tile[DT, N, IN_FLAT],
    output: Tile[DT, N, OUT_FLAT],
):
    var idx = Int(global_idx.x)
    if idx >= N * OUT_FLAT:
        return
    var b = idx // OUT_FLAT
    var out_pos = idx % OUT_FLAT
    var c = out_pos % C
    var sp = out_pos // C
    var oh = sp // OW
    var ow = sp % OW
    var best: Scalar[DT] = -3.0e38
    for kh in range(K):
        var ih = oh * S + kh - P
        if ih < 0 or ih >= H:
            continue
        for kw in range(K):
            var iw = ow * S + kw - P
            if iw < 0 or iw >= W:
                continue
            var v = rebind[Scalar[DT]](input[b, (ih * W + iw) * C + c])
            if v > best:
                best = v
    output[b, out_pos] = best


# host helper — fill one logical tensor into NCHW + NHWC buffers
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


def run_pool[
    N: Int, C: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    WARMUP: Int, ITERS: Int,
](ctx: DeviceContext, label: StaticString) raises:
    comptime OH = (H + 2 * P - K) // S + 1
    comptime OW = (W + 2 * P - K) // S + 1
    comptime IN_FLAT = C * H * W
    comptime OUT_FLAT = C * OH * OW
    comptime TOT = N * OUT_FLAT
    print(label, " POOL N=", N, " C=", C, " ", H, "x", W, " K=", K, " S=", S,
          " -> ", OH, "x", OW)

    var x_nchw = ctx.enqueue_create_buffer[DT](N * IN_FLAT)
    var x_nhwc = ctx.enqueue_create_buffer[DT](N * IN_FLAT)
    _fill_layouts[N, C, H, W, IN_FLAT](ctx, x_nchw, x_nhwc)
    var o_nchw = ctx.enqueue_create_buffer[DT](N * OUT_FLAT)
    var o_nhwc = ctx.enqueue_create_buffer[DT](N * OUT_FLAT)

    var xn = tile[N, IN_FLAT](x_nchw)
    var onn = tile[N, OUT_FLAT](o_nchw)
    var xh = tile[N, IN_FLAT](x_nhwc)
    var ohh = tile[N, OUT_FLAT](o_nhwc)
    comptime nb = (TOT + TPB - 1) // TPB

    ctx.enqueue_function[
        _maxpool_nchw[N, C, K, S, P, H, W, OH, OW, IN_FLAT, OUT_FLAT]
    ](xn, onn, grid_dim=nb, block_dim=TPB)
    ctx.enqueue_function[
        _maxpool_nhwc[N, C, K, S, P, H, W, OH, OW, IN_FLAT, OUT_FLAT]
    ](xh, ohh, grid_dim=nb, block_dim=TPB)
    ctx.synchronize()

    var osp = OH * OW
    var bad = 0
    with o_nchw.map_to_host() as hn:
        with o_nhwc.map_to_host() as hh:
            for n in range(N):
                for c in range(C):
                    for s in range(osp):
                        var a = hn[n * OUT_FLAT + c * osp + s]
                        var bb = hh[n * OUT_FLAT + s * C + c]
                        if abs(Float64(a - bb)) > 1e-4:
                            bad += 1
    print("  verify: out_mismatch=", bad)

    comptime for _ in range(WARMUP):
        ctx.enqueue_function[
            _maxpool_nchw[N, C, K, S, P, H, W, OH, OW, IN_FLAT, OUT_FLAT]
        ](xn, onn, grid_dim=nb, block_dim=TPB)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    comptime for _ in range(ITERS):
        ctx.enqueue_function[
            _maxpool_nchw[N, C, K, S, P, H, W, OH, OW, IN_FLAT, OUT_FLAT]
        ](xn, onn, grid_dim=nb, block_dim=TPB)
    ctx.synchronize()
    var us_n = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0

    comptime for _ in range(WARMUP):
        ctx.enqueue_function[
            _maxpool_nhwc[N, C, K, S, P, H, W, OH, OW, IN_FLAT, OUT_FLAT]
        ](xh, ohh, grid_dim=nb, block_dim=TPB)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    comptime for _ in range(ITERS):
        ctx.enqueue_function[
            _maxpool_nhwc[N, C, K, S, P, H, W, OH, OW, IN_FLAT, OUT_FLAT]
        ](xh, ohh, grid_dim=nb, block_dim=TPB)
    ctx.synchronize()
    var us_h = Float64(perf_counter_ns() - t1) / Float64(ITERS) / 1000.0

    print("  POOL NCHW: ", us_n, "us   POOL NHWC: ", us_h,
          "us   | NHWC/NCHW =", us_h / us_n, "x")


def main() raises:
    var ctx = DeviceContext()
    print("MaxPool2D NCHW-vs-NHWC parity [fp32]")
    print("=" * 70)
    run_pool[64, 32, 2, 2, 0, 48, 48, 5, 50](ctx, "rep48")
    run_pool[64, 64, 2, 2, 0, 24, 24, 5, 50](ctx, "rep24")
    run_pool[64, 32, 3, 2, 0, 84, 84, 5, 50](ctx, "atari84")
    print("=" * 70)
    print("GO if: all mismatches=0 AND NHWC/NCHW ~<=1.2x.")
    print("Perf truth = NVIDIA; Apple is parity-only.")
