"""Throughput of `sin_cos_device` (quat_math) against `std.math.sin` + `cos`
in a kernel — what the accurate NVIDIA path costs over `sin.approx`.

Each of N threads runs a dependent chain of ITERS sin/cos pairs
(`v = s + c`, which stays in [-1.42, 1.42]), so the work is not folded away
and both legs see the same argument range FK does. The checksum is printed
so the chain is live. Min of REPS timed launches after a warm-up.

Run: pixi run -e nvidia mojo run -I . benchmarks/bench_sin_cos_device.mojo
"""

from std.math import sin, cos
from std.time import perf_counter_ns
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from noeira.nn.core.ptr import mptr
from noeira.physics3d.kinematics.quat_math import sin_cos_device

comptime N = 1 << 20
comptime ITERS = 256
comptime REPS = 20


def _chain[ACCURATE: Bool](
    x: Pointer[Float32, MutAnyOrigin], out_: Pointer[Float32, MutAnyOrigin]
):
    var i = Int(global_idx.x)
    if i >= N:
        return
    var v = x[unsafe_offset=i]
    for _ in range(ITERS):
        var s: Float32
        var c: Float32
        comptime if ACCURATE:
            var sc = sin_cos_device(v)
            s = sc[0]
            c = sc[1]
        else:
            s = sin(v)
            c = cos(v)
        v = s + c
    out_[unsafe_offset=i] = v


def _time[ACCURATE: Bool](ctx: DeviceContext, mut dx: DeviceBuffer[DType.float32], mut dy: DeviceBuffer[DType.float32]) raises -> Float64:
    var best = Float64(1e30)
    for r in range(REPS + 1):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        ctx.enqueue_function[_chain[ACCURATE]](
            mptr(dx.unsafe_ptr()), mptr(dy.unsafe_ptr()),
            grid_dim=N // 256, block_dim=256,
        )
        ctx.synchronize()
        var dt = Float64(perf_counter_ns() - t0) / 1e6
        if r > 0 and dt < best:
            best = dt
    return best


def main() raises:
    var ctx = DeviceContext()
    var hx = ctx.enqueue_create_host_buffer[DType.float32](N)
    var hy = ctx.enqueue_create_host_buffer[DType.float32](N)
    var dx = ctx.enqueue_create_buffer[DType.float32](N)
    var dy = ctx.enqueue_create_buffer[DType.float32](N)
    ctx.synchronize()
    for i in range(N):
        hx[i] = Float32(-1.6 + 3.2 * Float64(i) / Float64(N))
    ctx.enqueue_copy(dx, hx)
    var t_b = _time[False](ctx, dx, dy)
    ctx.enqueue_copy(hy, dy)
    ctx.synchronize()
    var cs_b = Float64(0)
    for i in range(N):
        cs_b += Float64(hy[i])
    var t_a = _time[True](ctx, dx, dy)
    ctx.enqueue_copy(hy, dy)
    ctx.synchronize()
    var cs_a = Float64(0)
    for i in range(N):
        cs_a += Float64(hy[i])
    var pairs = Float64(N) * Float64(ITERS)
    print("sin+cos pairs per launch:", pairs)
    print("builtin   :", t_b, "ms ", pairs / t_b / 1e6, "Gpair/s  checksum", cs_b)
    print("accurate  :", t_a, "ms ", pairs / t_a / 1e6, "Gpair/s  checksum", cs_a)
    print("ratio accurate / builtin:", t_a / t_b)
