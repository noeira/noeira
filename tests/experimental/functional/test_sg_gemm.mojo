"""The simdgroup-matrix GEMM (`functional/sg_gemm.mojo`) against a CPU float64
reference — exits 1 on failure.

Every shape runs all eight (TA, TB, ACC) forms twice, launched through
metal_rt: through the tape's own dispatcher (`gpu._own_mm`: the direct
simdgroup kernels when M, N are tile multiples and K a multiple of 8, the
staged kernel otherwise, the naive kernel for small outputs), and through
the staged kernel forced (so it is gated on every shape too).
Shapes: multiples of the 64 tile, non-multiples of the tile and of 8 (edge
tiles in M, N and K), degenerate dimensions (1), and the tape's MLP shapes.

Bound: fp32 summation of K products has |error| <= gamma_K * sum_k |a_ik b_kj|
with gamma_K ~ K u (u = 2^-24). The gate allows (K + 2) u (sum |a b| + |o0|)
per element, where o0 is the value an ACC form adds onto. It reports the
worst error as a fraction of that bound and as a relative error.
"""

from std.random import seed, random_float64
from std.sys import exit

from noeira.experimental.functional import Tape
from noeira.experimental.functional.gpu import _own_mm
from noeira.experimental.functional.sg_gemm import sg_mm_kernel, SG_T, SG_THREADS

comptime F = DType.float32
comptime U: Float64 = 5.960464477539063e-08


def run[TA: Bool, TB: Bool, ACC: Bool, STAGED: Bool](
    tape: Tape, m: Int, n: Int, k: Int, mut worst_bound: Float64,
    mut worst_rel: Float64,
) raises -> Bool:
    """STAGED: force the staged kernel; else the tape's own dispatcher
    (`gpu._own_mm`: direct simdgroup kernels, staged, or naive)."""
    var d = tape.data[].dctx()
    var ha = List[Float32](length=m * k, fill=0)
    var hb = List[Float32](length=k * n, fill=0)
    var o0 = List[Float32](length=m * n, fill=0)
    for i in range(m * k):
        ha[i] = Float32(random_float64() * 2.0 - 1.0)
    for i in range(k * n):
        hb[i] = Float32(random_float64() * 2.0 - 1.0)
    for i in range(m * n):
        o0[i] = Float32(random_float64() * 2.0 - 1.0)
    var a = tape.device_buffer(m * k)
    var b = tape.device_buffer(k * n)
    var o = tape.device_buffer(m * n)
    tape.upload(a, ha)
    tape.upload(b, hb)
    tape.upload(o, o0)
    comptime if STAGED:
        d.enqueue_function[sg_mm_kernel[TA, TB, ACC]](
            o, a, b, Int64(m), Int64(n), Int64(k),
            grid_dim=((n + SG_T - 1) // SG_T, (m + SG_T - 1) // SG_T),
            block_dim=(SG_THREADS, 1),
        )
    else:
        _own_mm[TA, TB, ACC](d, o, a, b, m, n, k)
    var ho = List[Float32](length=m * n, fill=0)
    d.enqueue_copy(ho.unsafe_ptr(), o)
    var ok = True
    for i in range(m):
        for j in range(n):
            var s: Float64 = 0
            var sa: Float64 = 0
            for kk in range(k):
                var av = Float64(ha[(kk * m + i) if TA else (i * k + kk)])
                var bv = Float64(hb[(j * k + kk) if TB else (kk * n + j)])
                s += av * bv
                sa += abs(av * bv)
            comptime if ACC:
                s += Float64(o0[i * n + j])
                sa += abs(Float64(o0[i * n + j]))
            var err = abs(Float64(ho[i * n + j]) - s)
            var bound = Float64(k + 2) * U * sa + 1e-30
            worst_bound = max(worst_bound, err / bound)
            if abs(s) > 1e-3:
                worst_rel = max(worst_rel, err / abs(s))
            if err > bound:
                ok = False
    return ok


def forms[STAGED: Bool](
    tape: Tape, m: Int, n: Int, k: Int, mut wb: Float64, mut wr: Float64
) raises -> Bool:
    var ok = True
    ok = run[False, False, False, STAGED](tape, m, n, k, wb, wr) and ok
    ok = run[False, False, True, STAGED](tape, m, n, k, wb, wr) and ok
    ok = run[True, False, False, STAGED](tape, m, n, k, wb, wr) and ok
    ok = run[True, False, True, STAGED](tape, m, n, k, wb, wr) and ok
    ok = run[False, True, False, STAGED](tape, m, n, k, wb, wr) and ok
    ok = run[False, True, True, STAGED](tape, m, n, k, wb, wr) and ok
    ok = run[True, True, False, STAGED](tape, m, n, k, wb, wr) and ok
    ok = run[True, True, True, STAGED](tape, m, n, k, wb, wr) and ok
    return ok


def main() raises:
    seed(11)
    var tape = Tape.metal_rt()
    var shapes = List[List[Int]]()
    shapes.append([64, 64, 64])
    shapes.append([128, 192, 64])
    shapes.append([256, 256, 256])
    shapes.append([65, 63, 17])
    shapes.append([100, 37, 5])
    shapes.append([7, 9, 3])
    shapes.append([1, 64, 64])
    shapes.append([64, 1, 64])
    shapes.append([64, 64, 1])
    shapes.append([1, 1, 1])
    shapes.append([256, 23, 256])
    shapes.append([300, 513, 70])
    shapes.append([512, 512, 128])
    shapes.append([256, 128, 512])
    shapes.append([64, 512, 256])
    shapes.append([512, 64, 40])
    var fails = 0
    var cases = 0
    var wb: Float64 = 0
    var wr: Float64 = 0
    for s in range(len(shapes)):
        var m = shapes[s][0]
        var n = shapes[s][1]
        var k = shapes[s][2]
        var wb0 = wb
        wb = 0
        var ok1 = forms[False](tape, m, n, k, wb, wr)
        var w1 = wb
        wb = 0
        var ok2 = forms[True](tape, m, n, k, wb, wr)
        var ok = ok1 and ok2
        cases += 16
        print("  [", m, "x", k, "] @ [", k, "x", n, "], 8 forms: dispatcher err/bound", w1,
              "| staged kernel err/bound", wb, "OK" if ok else "FAIL")
        wb = max(max(wb, w1), wb0)
        if not ok:
            fails += 1
    print("worst error / bound:", wb, "| worst relative error (|ref| > 1e-3):", wr)
    print("GEMM GATE:", cases, "cases,", fails, "shapes failed")
    if fails != 0:
        exit(1)
    print("ALL OK")
