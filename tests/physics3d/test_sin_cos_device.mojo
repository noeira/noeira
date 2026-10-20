"""`sin_cos_device` (quat_math) in a kernel against a correctly rounded
reference: float64 libm on the host, rounded to float32.

CUDA lowers float32 `std.math.sin` / `cos` to `sin.approx` / `cos.approx`
(3.3e-7 absolute on an RTX 5090, 1e-5 relative at small angles), and that
alone failed `test_euler_finalize_gpu_parity` on NVIDIA. This gates the
replacement where it is used, in a kernel:

  [1] 8191 angles on [-1.6, 1.6] (the joint half-angles FK sees): absolute
      error <= 1.5e-7 on sin and cos (2.5 float32 ulps at 1.0; the builtin
      reads 3.3e-7 on the 5090)
  [2] relative error of sin at small angles (1e-4 .. 0.1) <= 2.5e-7 (~2 ulps); the
      builtin reads 1e-5 at 0.025
  [3] 4096 angles on [-200, 200] (a spinning wheel's continuous hinge):
      absolute error <= 3e-7, every quadrant crossed
  [4] the quarter turns: sin(k pi/2), cos(k pi/2) to 3e-7 for k in -8..8

On Apple / CPU the function IS `std.math`, so this then gates nothing new;
it is in the GPU manifest for the NVIDIA run.

Run: pixi run -e nvidia mojo run -I . tests/physics3d/test_sin_cos_device.mojo
"""

from std.math import sin, cos, abs
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from noeira.nn.core.ptr import mptr
from noeira.physics3d.kinematics.quat_math import sin_cos_device

comptime N_A = 8191
comptime N_B = 1000
comptime N_C = 4096
comptime N_D = 17
comptime N = N_A + N_B + N_C + N_D


def _kernel(
    x: Pointer[Float32, MutAnyOrigin],
    s: Pointer[Float32, MutAnyOrigin],
    c: Pointer[Float32, MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        var sc = sin_cos_device(x[unsafe_offset=i])
        s[unsafe_offset=i] = sc[0]
        c[unsafe_offset=i] = sc[1]


def main() raises:
    var ctx = DeviceContext()
    var hx = ctx.enqueue_create_host_buffer[DType.float32](N)
    var hs = ctx.enqueue_create_host_buffer[DType.float32](N)
    var hc = ctx.enqueue_create_host_buffer[DType.float32](N)
    var dx = ctx.enqueue_create_buffer[DType.float32](N)
    var ds = ctx.enqueue_create_buffer[DType.float32](N)
    var dc = ctx.enqueue_create_buffer[DType.float32](N)
    ctx.synchronize()
    var k = 0
    for i in range(N_A):
        hx[k] = Float32(-1.6 + 3.2 * Float64(i) / Float64(N_A - 1))
        k += 1
    for i in range(N_B):
        hx[k] = Float32(1e-4 * (1000.0 ** (Float64(i) / Float64(N_B - 1))))
        k += 1
    for i in range(N_C):
        hx[k] = Float32(-200.0 + 400.0 * Float64(i) / Float64(N_C - 1))
        k += 1
    for i in range(N_D):
        hx[k] = Float32(Float64(i - 8) * 1.5707963267948966)
        k += 1
    ctx.enqueue_copy(dx, hx)
    ctx.enqueue_function[_kernel](
        mptr(dx.unsafe_ptr()), mptr(ds.unsafe_ptr()), mptr(dc.unsafe_ptr()),
        grid_dim=(N + 255) // 256, block_dim=256,
    )
    ctx.enqueue_copy(hs, ds)
    ctx.enqueue_copy(hc, dc)
    ctx.synchronize()

    var fails = 0

    def _worst_abs(lo: Int, hi: Int) capturing -> Float64:
        var w = Float64(0)
        for i in range(lo, hi):
            var x = Float64(hx[i])
            w = max(w, abs(Float64(hs[i]) - Float64(Float32(sin(x)))))
            w = max(w, abs(Float64(hc[i]) - Float64(Float32(cos(x)))))
        return w

    var a = _worst_abs(0, N_A)
    print("[1] |x| <= 1.6: worst abs err", a, "(bound 1.5e-7)")
    if a > 1.5e-7:
        fails += 1

    var r = Float64(0)
    for i in range(N_A, N_A + N_B):
        var x = Float64(hx[i])
        var ref_ = Float64(Float32(sin(x)))
        r = max(r, abs(Float64(hs[i]) - ref_) / abs(ref_))
    print("[2] 1e-4 <= x <= 0.1: worst rel err of sin", r, "(bound 2.5e-7)")
    if r > 2.5e-7:
        fails += 1

    var b = _worst_abs(N_A + N_B, N_A + N_B + N_C)
    print("[3] |x| <= 200: worst abs err", b, "(bound 3e-7)")
    if b > 3e-7:
        fails += 1

    var d = _worst_abs(N_A + N_B + N_C, N)
    print("[4] k pi/2, k in -8..8: worst abs err", d, "(bound 3e-7)")
    if d > 3e-7:
        fails += 1

    if fails > 0:
        raise Error(String(fails, " leg(s) out of bound"))
    print("test_sin_cos_device: ALL PASS")
