"""PCSwish unit test — numeric sanity + CPU/GPU parity.

Validates `PCSwish.apply` and `PCSwish.apply_derivative_mul` against
hand-computed reference values and confirms the GPU kernels match the
CPU path bitwise (within float32 noise).

Reference math: f(x) = x · σ(x); f'(x) = σ(x) · (1 + x · (1 − σ(x))).
σ(x) = 1 / (1 + exp(-x)). At x=0 → σ=0.5, f=0, f'=0.5.

Run:
    pixi run -e apple  mojo run -I . tests/pcn/test_pc_swish.mojo
    pixi run -e nvidia mojo run -I . tests/pcn/test_pc_swish.mojo
"""

from std.math import abs as mabs
from max.gpu.host import DeviceContext
from noeira.core.tiles import tile

from noeira.experimental.pcn import PCSwish
from noeira.nn.core.ptr import mptr


comptime BATCH = 1
comptime DIM = 5
comptime TOL: Float64 = 1.0e-5


def main() raises:
    print("=" * 60)
    print("PCSwish unit test")
    print("=" * 60)

    # Test inputs spanning sign + magnitude. Reference values computed in
    # numpy (float32):
    #   f(-2) = -2·σ(-2) = -0.23840584; f'(-2) = -0.09078424
    #   f(-1) = -1·σ(-1) = -0.26894143; f'(-1) =  0.07232950
    #   f( 0) =  0;                    f'( 0) =  0.5
    #   f( 1) =  1·σ( 1) =  0.7310586; f'( 1) =  0.9276705
    #   f( 2) =  2·σ( 2) =  1.7615942; f'( 2) =  1.0907842
    var x_vals = List[Float32](capacity=DIM)
    x_vals.append(-2.0)
    x_vals.append(-1.0)
    x_vals.append(0.0)
    x_vals.append(1.0)
    x_vals.append(2.0)
    var f_ref = List[Float64](capacity=DIM)
    f_ref.append(-0.23840584)
    f_ref.append(-0.26894143)
    f_ref.append(0.0)
    f_ref.append(0.7310586)
    f_ref.append(1.7615942)
    var fp_ref = List[Float64](capacity=DIM)
    fp_ref.append(-0.09078424)
    fp_ref.append(0.07232950)
    fp_ref.append(0.5)
    fp_ref.append(0.9276705)
    fp_ref.append(1.0907842)

    # ── CPU path ─────────────────────────────────────────────────────────────
    var x_buf = List[Float32](capacity=BATCH * DIM)
    for i in range(DIM):
        x_buf.append(x_vals[i])
    var a_buf = List[Float32](capacity=BATCH * DIM)
    for _ in range(BATCH * DIM):
        a_buf.append(0.0)
    var z_in_buf = List[Float32](capacity=BATCH * DIM)
    for _ in range(BATCH * DIM):
        z_in_buf.append(1.0)
    var z_out_buf = List[Float32](capacity=BATCH * DIM)
    for _ in range(BATCH * DIM):
        z_out_buf.append(0.0)

    var x_t = tile[BATCH, DIM](mptr(x_buf))
    var a_t = tile[BATCH, DIM](mptr(a_buf))
    var z_in_t = tile[BATCH, DIM](mptr(z_in_buf))
    var z_out_t = tile[BATCH, DIM](mptr(z_out_buf))

    PCSwish.apply[BATCH, DIM, DType.float32](x_t, a_t)
    PCSwish.apply_derivative_mul[BATCH, DIM, DType.float32](x_t, z_in_t, z_out_t)

    print("\nCPU vs reference:")
    var max_err_cpu: Float64 = 0.0
    for i in range(DIM):
        var f = Float64(a_buf[i])
        var fp = Float64(z_out_buf[i])
        var ef = mabs(f - f_ref[i])
        var efp = mabs(fp - fp_ref[i])
        if ef > max_err_cpu:
            max_err_cpu = ef
        if efp > max_err_cpu:
            max_err_cpu = efp
        print(
            "  x =", x_buf[i],
            " f =", f, " (ref", f_ref[i], ", err", ef, ")",
            " f'=", fp, " (ref", fp_ref[i], ", err", efp, ")",
        )
    print("CPU max abs err:", max_err_cpu, " (tol", TOL, ")")
    if max_err_cpu > TOL:
        print("FAIL: CPU values diverge from reference")
        return

    # ── GPU path ─────────────────────────────────────────────────────────────
    print("\nGPU parity:")
    with DeviceContext() as ctx:
        var x_dbuf = ctx.enqueue_create_buffer[DType.float32](BATCH * DIM)
        var a_dbuf = ctx.enqueue_create_buffer[DType.float32](BATCH * DIM)
        var z_in_dbuf = ctx.enqueue_create_buffer[DType.float32](BATCH * DIM)
        var z_out_dbuf = ctx.enqueue_create_buffer[DType.float32](BATCH * DIM)
        var x_host = ctx.enqueue_create_host_buffer[DType.float32](BATCH * DIM)
        var z_in_host = ctx.enqueue_create_host_buffer[DType.float32](BATCH * DIM)
        var a_host = ctx.enqueue_create_host_buffer[DType.float32](BATCH * DIM)
        var z_out_host = ctx.enqueue_create_host_buffer[DType.float32](BATCH * DIM)
        for i in range(BATCH * DIM):
            x_host.unsafe_ptr()[i] = x_buf[i]
            z_in_host.unsafe_ptr()[i] = 1.0
        ctx.enqueue_copy(x_dbuf, x_host)
        ctx.enqueue_copy(z_in_dbuf, z_in_host)
        ctx.synchronize()

        var x_d_t = tile[BATCH, DIM](x_dbuf.unsafe_ptr().as_unsafe_any_origin())
        var a_d_t = tile[BATCH, DIM](a_dbuf.unsafe_ptr().as_unsafe_any_origin())
        var z_in_d_t = tile[BATCH, DIM](z_in_dbuf.unsafe_ptr().as_unsafe_any_origin())
        var z_out_d_t = tile[BATCH, DIM](z_out_dbuf.unsafe_ptr().as_unsafe_any_origin())

        PCSwish.apply_gpu[BATCH, DIM, DType.float32](ctx, x_d_t, a_d_t)
        PCSwish.apply_derivative_mul_gpu[BATCH, DIM, DType.float32](
            ctx, x_d_t, z_in_d_t, z_out_d_t
        )
        ctx.enqueue_copy(a_host, a_dbuf)
        ctx.enqueue_copy(z_out_host, z_out_dbuf)
        ctx.synchronize()

        var max_err_gpu: Float64 = 0.0
        for i in range(BATCH * DIM):
            var ef = mabs(Float64(a_host.unsafe_ptr()[i]) - Float64(a_buf[i]))
            var efp = mabs(
                Float64(z_out_host.unsafe_ptr()[i]) - Float64(z_out_buf[i])
            )
            if ef > max_err_gpu:
                max_err_gpu = ef
            if efp > max_err_gpu:
                max_err_gpu = efp
        print("GPU vs CPU max abs diff:", max_err_gpu, " (tol", TOL, ")")
        if max_err_gpu > TOL:
            # ⚠ RAISE, NEVER `return`: `scripts/run_tests.sh` reads the EXIT
            # CODE and nothing else, so a printed FAIL followed by a normal
            # return is reported by the manifest as a PASS.
            raise (
                String("pc_swish: GPU diverges from CPU by ")
                + String(max_err_gpu)
            )

    print("\n=== PASS ===")
