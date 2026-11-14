# +--------------------------------------------------------------------------+ #
# | 4-bit against 8-bit weights: one decoded token's GEMVs, timed
# +--------------------------------------------------------------------------+ #
"""Is a token faster to decode from 4-bit weights than from int8, on this GPU?
(docs/NN_MODEL_PORTS_PLAN.md §12.8, S2: asked before the 4-bit model is built.)

    pixi run -e nvidia mojo build -I . benchmarks/bench_q4_gemv_gpu.mojo -o build/bench_q4
    ./build/bench_q4 [reps]

1. `q4_gemv_kernel` against a host computation, on a small matrix with every
   4-bit value and a scale per block: the kernel times the right thing.
2. One decoded token's linears of LFM3-1.7B (153 weights, 1.466 G values:
   the conv, attention and MLP projections of its 24 layers, each its own
   memory), as GEMVs for 1, 2 and 4 rows of activations: `q8_gemv_kernel` on
   int8 (1.47 GB) and `q4_gemv_kernel` on Q4_0 (0.73 GB + 0.18 GB of scales).
   The values are constants: the time does not depend on them.

No model is loaded. Needs 2.5 GB on the device.
"""

from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.llm.lfm3.int8_kernels import q8_gemv_kernel, Q8_RPW
from noeira.llm.lfm3.q4_kernels import q4_gemv_kernel, Q4_RPW, Q4_BLOCK


def ms(t0: Int) -> Float64:
    return Float64(perf_counter_ns() - t0) / 1e6


def median(var xs: List[Float64]) -> Float64:
    for i in range(1, len(xs)):
        var j = i
        while j > 0 and xs[j - 1] > xs[j]:
            var t = xs[j]
            xs[j] = xs[j - 1]
            xs[j - 1] = t
            j -= 1
    return xs[len(xs) // 2]


def check_small(ctx: DeviceContext) raises -> Float64:
    """The kernel on `n = 8, k = 96, NS = 2` against the host: max |d| / max |y|."""
    var n = 8
    var k = 96
    var kb = k // 2
    var nb = k // Q4_BLOCK
    var hq = ctx.enqueue_create_host_buffer[DType.uint8](n * kb)
    var hd = ctx.enqueue_create_host_buffer[DT](n * nb)
    var hx = ctx.enqueue_create_host_buffer[DT](2 * k)
    ctx.synchronize()
    for i in range(n * kb):
        hq[i] = UInt8((i * 37 + 11) % 256)
    for i in range(n * nb):
        hd[i] = Float32(0.01) * Float32(1 + i % 7) * (Float32(-1.0) if i % 3 == 0 else Float32(1.0))
    for i in range(2 * k):
        hx[i] = Float32((i * 13) % 17) / Float32(8.0) - Float32(1.0)
    var want = List[Float32]()
    for s in range(2):
        for r in range(n):
            var acc = Float64(0.0)
            for b in range(nb):
                var part = Float64(0.0)
                for i in range(16):
                    var by = Int(hq[r * kb + b * 16 + i])
                    part += Float64((by & 15) - 8) * Float64(hx[s * k + b * 32 + i])
                    part += Float64((by >> 4) - 8) * Float64(hx[s * k + b * 32 + 16 + i])
                acc += Float64(hd[r * nb + b]) * part
            want.append(Float32(acc))
    var dq = ctx.enqueue_create_buffer[DType.uint8](n * kb)
    var dd = ctx.enqueue_create_buffer[DT](n * nb)
    var dx = ctx.enqueue_create_buffer[DT](2 * k)
    var dy = ctx.enqueue_create_buffer[DT](2 * n)
    ctx.enqueue_copy(dq, hq)
    ctx.enqueue_copy(dd, hd)
    ctx.enqueue_copy(dx, hx)
    ctx.enqueue_function[q4_gemv_kernel[2]](dx, dq, dd, dy, Int64(n), Int64(k), grid_dim=1, block_dim=256)
    var hy = ctx.enqueue_create_host_buffer[DT](2 * n)
    ctx.enqueue_copy(hy, dy)
    ctx.synchronize()
    var top = Float64(0.0)
    var err = Float64(0.0)
    for i in range(2 * n):
        top = max(top, abs(Float64(want[i])))
        err = max(err, abs(Float64(hy[i]) - Float64(want[i])))
    return err / top


def main() raises:
    var args = argv()
    var reps = Int(String(args[1])) if len(args) > 1 else 20
    var ctx = DeviceContext()
    var e = check_small(ctx)
    print("q4_gemv_kernel against the host (8 x 96, 2 rows): max |d| / max |y| =", e, "PASS" if e < 1e-5 else "FAIL")
    if e >= 1e-5:
        raise Error("the 4-bit kernel is wrong: nothing is timed")

    # one token's linears, in the order a layer applies them: [rows, columns]
    var ns = List[Int]()
    var ks = List[Int]()
    var kinds = "ccFcScFcScFcScFcScFcScFc"
    for ch in kinds.as_bytes():
        if ch == 99:  # c: the short conv
            ns.append(6144)
            ks.append(2048)
            ns.append(2048)
            ks.append(2048)
        else:
            var an: List[Int] = [2048, 512, 512, 16, 2048]  # q, k, v, gate, o
            for a in an:
                ns.append(a)
                ks.append(2048)
        ns.append(7680)  # w1
        ks.append(2048)
        ns.append(7680)  # w3
        ks.append(2048)
        ns.append(2048)  # w2
        ks.append(7680)
    var total = 0
    var rows = 0
    for i in range(len(ns)):
        total += ns[i] * ks[i]
        rows += ns[i]
    print(len(ns), "weights,", total, "values:", total // 1000000, "MB int8,", total // 2000000, "MB of 4-bit values +",
          total // 8000000, "MB of scales")
    var q8 = ctx.enqueue_create_buffer[DType.int8](total)
    var s8 = ctx.enqueue_create_buffer[DT](rows)
    var q4 = ctx.enqueue_create_buffer[DType.uint8](total // 2)
    var d4 = ctx.enqueue_create_buffer[DT](total // Q4_BLOCK)
    q8.enqueue_fill(Int8(3))
    s8.enqueue_fill(Float32(0.01))
    q4.enqueue_fill(UInt8(0x5A))
    d4.enqueue_fill(Float32(0.01))
    var x = ctx.enqueue_create_buffer[DT](4 * 7680)
    var y = ctx.enqueue_create_buffer[DT](4 * 7680)
    x.enqueue_fill(Float32(0.5))
    ctx.synchronize()

    for t in [1, 2, 4]:
        var t8 = List[Float64]()
        var t4 = List[Float64]()
        for r in range(reps + 2):
            var off = 0
            var roff = 0
            var t0 = perf_counter_ns()
            for i in range(len(ns)):
                var n = ns[i]
                var k = ks[i]
                var q = q8.create_sub_buffer[DType.int8](off, n * k)
                var sc = s8.create_sub_buffer[DT](roff, n)
                var g = (n + 8 * Q8_RPW - 1) // (8 * Q8_RPW)
                if t == 1:
                    ctx.enqueue_function[q8_gemv_kernel[1]](x, q, sc, y, Int64(n), Int64(k), grid_dim=g, block_dim=256)
                elif t == 2:
                    ctx.enqueue_function[q8_gemv_kernel[2]](x, q, sc, y, Int64(n), Int64(k), grid_dim=g, block_dim=256)
                else:
                    ctx.enqueue_function[q8_gemv_kernel[4]](x, q, sc, y, Int64(n), Int64(k), grid_dim=g, block_dim=256)
                off += n * k
                roff += n
            ctx.synchronize()
            if r >= 2:
                t8.append(ms(t0))
            off = 0
            var t1 = perf_counter_ns()
            for i in range(len(ns)):
                var n = ns[i]
                var k = ks[i]
                var q = q4.create_sub_buffer[DType.uint8](off // 2, n * k // 2)
                var dd = d4.create_sub_buffer[DT](off // Q4_BLOCK, n * k // Q4_BLOCK)
                var g = (n + 8 * Q4_RPW - 1) // (8 * Q4_RPW)
                if t == 1:
                    ctx.enqueue_function[q4_gemv_kernel[1]](x, q, dd, y, Int64(n), Int64(k), grid_dim=g, block_dim=256)
                elif t == 2:
                    ctx.enqueue_function[q4_gemv_kernel[2]](x, q, dd, y, Int64(n), Int64(k), grid_dim=g, block_dim=256)
                else:
                    ctx.enqueue_function[q4_gemv_kernel[4]](x, q, dd, y, Int64(n), Int64(k), grid_dim=g, block_dim=256)
                off += n * k
            ctx.synchronize()
            if r >= 2:
                t4.append(ms(t1))
        var m8 = median(t8^)
        var m4 = median(t4^)
        print("  ", t, "rows of activations: int8", round(m8, 2), "ms, 4-bit", round(m4, 2), "ms (", round(m8 / m4, 2), "x )")
