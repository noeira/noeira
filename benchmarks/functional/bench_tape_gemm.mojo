"""The functional tape's GEMMs on Metal: the old tiled kernel, the
simdgroup-matrix kernel (`functional/sg_gemm.mojo`) and MAX's matmul (black
box), TFLOP/s on the shapes of the B=4096 64-512-512-64 MLP step.

Columns: the old scalar tiled kernel; the staged simdgroup kernel (edge
tiles, any K) forced; `sg_mm`, what the tape runs (the direct kernel for
the form when M, N are tile multiples and K a multiple of 8); MAX.

Forward `x @ w`, and the two backward products: dX = dO @ wᵀ (TB) and
dW = xᵀ @ dO (TA). MAX's matmul has no transpose_a: the tape runs dW on MAX
as a transpose kernel + matmul, and so does this bench (that is what the
tape pays). Own kernels launch through metal_rt, MAX through MAX. Each cell:
20 launches then a sync, best of 5.

    pixi run -e apple mojo build -I . benchmarks/functional/bench_tape_gemm.mojo -o /tmp/bench_gemm && /tmp/bench_gemm
"""

from std.time import perf_counter_ns
from std.random import seed, random_float64
from max.gpu.host import DeviceBuffer

from noeira.nn.constants import DT
from noeira.experimental.functional import Tape
from noeira.experimental.functional.backend import Dev
from noeira.experimental.functional.gpu import (
    _tiled_mm_kernel, _max_mm, _naive_mm, gpu_transpose, MM_T, MM_SUB,
)
from noeira.experimental.functional.sg_gemm import sg_mm, sg_mm_kernel, SG_T, SG_THREADS
from max.gpu.host import DeviceContext

comptime REPS = 20


def _rnd(n: Int) -> List[Scalar[DT]]:
    var o = List[Scalar[DT]]()
    for _ in range(n):
        o.append(Scalar[DT](random_float64() - 0.5))
    return o^


def _launch[ARM: Int, TA: Bool, TB: Bool](
    d: Dev, o: DeviceBuffer[DT], a: DeviceBuffer[DT], b: DeviceBuffer[DT],
    t: DeviceBuffer[DT], m: Int, n: Int, k: Int,
) raises:
    comptime if ARM == 0:
        d.enqueue_function[_tiled_mm_kernel[TA, TB, False]](
            o, a, b, Int64(m), Int64(n), Int64(k),
            grid_dim=((n + MM_T - 1) // MM_T, (m + MM_T - 1) // MM_T),
            block_dim=(MM_SUB, MM_SUB),
        )
    elif ARM == 1:
        sg_mm[TA, TB, False](d, o, a, b, m, n, k)
    elif ARM == 4:
        _naive_mm[TA, TB, False](d, o, a, b, m, n, k)
    elif ARM == 3:
        d.enqueue_function[sg_mm_kernel[TA, TB, False]](
            o, a, b, Int64(m), Int64(n), Int64(k),
            grid_dim=((n + SG_T - 1) // SG_T, (m + SG_T - 1) // SG_T),
            block_dim=(SG_THREADS, 1),
        )
    else:
        comptime if TA:
            gpu_transpose(d, t, a, k, m)  # a is [k, m] -> t [m, k]
            _max_mm[transpose_b=TB](d, o, t, b, m, n, k)
        else:
            _max_mm[transpose_b=TB](d, o, a, b, m, n, k)


def _time[ARM: Int, TA: Bool, TB: Bool](
    tape: Tape, m: Int, n: Int, k: Int
) raises -> Float64:
    var a = tape.device_buffer(m * k)
    var b = tape.device_buffer(k * n)
    var o = tape.device_buffer(m * n)
    var t = tape.device_buffer(m * k)
    tape.upload(a, _rnd(m * k))
    tape.upload(b, _rnd(k * n))
    var d = tape.data[].dctx()
    var best = Float64(1e30)
    for r in range(6):
        tape.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(REPS):
            _launch[ARM, TA, TB](d, o, a, b, t, m, n, k)
        tape.synchronize()
        var us = Float64(perf_counter_ns() - t0) / 1e3 / Float64(REPS)
        if r > 0:
            best = min(best, us)
    return 2.0 * Float64(m) * Float64(n) * Float64(k) / best / 1e6


def _row[TA: Bool, TB: Bool](
    rt: Tape, mx: Tape, name: String, m: Int, n: Int, k: Int
) raises:
    var old = _time[0, TA, TB](rt, m, n, k)
    var sg = _time[1, TA, TB](rt, m, n, k)
    var mm = _time[2, TA, TB](mx, m, n, k)
    var st = _time[3, TA, TB](rt, m, n, k)
    var nv = _time[4, TA, TB](rt, m, n, k)
    print("|", name, "|", m, "x", n, "x", k, "|", _f(nv), "|", _f(old), "|", _f(st), "|",
          _f(sg), "|", _f(mm), "|", _f(sg / mm), "|")


def _f(x: Float64) -> String:
    var r = Int(x * 100.0 + 0.5)
    var c = r % 100
    return String(r // 100) + "." + (String("0") if c < 10 else String("")) + String(c)


def main() raises:
    seed(5)
    var rt = Tape.metal_rt()
    var mx = Tape(DeviceContext())
    print("TFLOP/s (2·M·N·K / time), M1 Pro")
    print("| product | M x N x K | naive | old tiled | simdgroup staged | simdgroup sg_mm | MAX matmul | sg_mm / MAX |")
    print("|---|---|---|---|---|---|---|---|")
    _row[False, False](rt, mx, "L1 fwd x@w", 4096, 512, 64)
    _row[False, False](rt, mx, "L2 fwd x@w", 4096, 512, 512)
    _row[False, False](rt, mx, "L3 fwd x@w", 4096, 64, 512)
    _row[False, True](rt, mx, "L3 dX = dO@wT", 4096, 512, 64)
    _row[False, True](rt, mx, "L2 dX = dO@wT", 4096, 512, 512)
    _row[True, False](rt, mx, "L3 dW = xT@dO", 512, 64, 4096)
    _row[True, False](rt, mx, "L2 dW = xT@dO", 512, 512, 4096)
    _row[True, False](rt, mx, "L1 dW = xT@dO", 64, 512, 4096)
    _row[False, False](rt, mx, "SAC fwd", 256, 256, 256)
    _row[False, False](rt, mx, "SAC L1 fwd", 256, 256, 23)
    _row[True, False](rt, mx, "SAC dW", 256, 256, 256)
    _row[False, False](rt, mx, "PPO fwd", 64, 64, 64)
    _row[True, False](rt, mx, "PPO dW", 64, 64, 64)
    _row[False, False](rt, mx, "PPO L1 fwd", 64, 64, 8)
    _row[False, False](rt, mx, "128 x 128 x 128", 128, 128, 128)
