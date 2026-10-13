"""`gptq_q4_0` end to end (the inverse-Hessian factor plus the column loop)
at SmolVLA's text shapes and at Kev-9B's largest.

    pixi run mojo run -I . benchmarks/quant/bench_gptq.mojo [--kev]

Shapes are [OUT, IN] (the Hessian is IN x IN): SmolVLA's q / o 960 x 960,
k / v 320 x 960, gate / up 2560 x 960, down 960 x 2560; with `--kev`, Kev-9B's
down_proj 4096 x 12288. H is built from IN / 2 random rows (rank deficient
before the damping, as a calibration Hessian is); its build is not timed.
Also prints GPTQ's objective against RTN's on that H.
"""

from std.random import random_float64, seed
from std.sys import argv
from std.time import perf_counter_ns

from noeira.nn.quant.dense import gemm_acc
from noeira.nn.quant.q4_0 import quantize_rtn_q4_0
from noeira.nn.quant.gptq import gptq_q4_0, q4_0_objective


def _p(mut v: List[Float64]) -> Pointer[Float64, MutAnyOrigin]:
    return rebind[Pointer[Float64, MutAnyOrigin]](v.unsafe_ptr())


def hessian(k: Int) -> List[Float64]:
    var t = k // 2
    var xt = List[Float64](length=k * t, fill=0)
    for i in range(k):
        var s = 1.0 + 29.0 * random_float64()
        for r in range(t):
            xt[i * t + r] = s * (random_float64() - 0.5)
    var x = List[Float64](length=t * k, fill=0)
    for i in range(k):
        for r in range(t):
            x[r * k + i] = xt[i * t + r]
    var h = List[Float64](length=k * k, fill=0)
    gemm_acc[DType.float64](k, k, t, 1.0, _p(xt), t, _p(x), k, _p(h), k)
    return h^


def one(name: String, n: Int, k: Int) raises:
    var h = hessian(k)
    var w = List[Float32](capacity=n * k)
    for _ in range(n * k):
        w.append(Float32((random_float64() - 0.5) * 0.1))
    var q = List[UInt8]()
    var d = List[Float32]()
    var t0 = perf_counter_ns()
    _ = gptq_q4_0(w, n, k, h.copy(), q, d)
    var sec = Float64(perf_counter_ns() - t0) / 1e9
    var qr = List[UInt8]()
    var dr = List[Float32]()
    quantize_rtn_q4_0(w, n, k, qr, dr)
    var og = q4_0_objective(w, q, d, n, k, h)
    var orr = q4_0_objective(w, qr, dr, n, k, h)
    print("  ", name, "[", n, "x", k, "]  gptq_q4_0", sec, "s   objective GPTQ / RTN", og / orr)


def main() raises:
    seed(9)
    var kev = False
    var args = argv()
    for i in range(1, len(args)):
        if String(args[i]) == "--kev":
            kev = True
    one("q / o   ", 960, 960)
    one("k / v   ", 320, 960)
    one("gate/up ", 2560, 960)
    one("down    ", 960, 2560)
    if kev:
        one("kev down", 4096, 12288)
