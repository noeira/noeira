"""ConvPCBlock CPU-vs-GPU parity (P2) — see docs/PCN_CONV_DESIGN.md.

Runs ConvPCBlock's GPU dispatchers (predict_gpu / pull_back_gpu /
weight_grad_gpu, plus the ACT-driven a_below) and checks they match the CPU
naive implementations. CPU-vs-GPU parity is the right check here (not finite
differences — see feedback_fd_gradcheck_tf32: TF32 ULP quantization fakes
kernel bugs).

The GPU kernels use atomic-free GATHER forms, so on a deterministic input the
GPU result should match the CPU loops to float32 rounding.

Run (Apple):
    pixi run -e apple mojo run -I . tests/pcn/test_conv_pc_block_gpu_parity.mojo
Run (NVIDIA):
    pixi run -e nvidia mojo run -I . tests/pcn/test_conv_pc_block_gpu_parity.mojo
"""

from std.memory import alloc
from std.math import sin
from noeira.core.tiles import tile
from max.gpu.host import DeviceContext

from noeira.experimental.pcn.pc_initializer import PCXavier
from noeira.experimental.pcn.pc_conv_block import ConvPCBlock
from noeira.experimental.pcn import PCReLU

comptime dtype = DType.float32
comptime TOL: Float32 = 1e-3


def _max_abs_diff(
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    b: Pointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
) -> Float32:
    var m: Float32 = 0.0
    for i in range(n):
        var d = a[i] - b[i]
        var ad = d if d >= 0 else -d
        if ad > m:
            m = ad
    return m


def run_gpu_parity[
    IC: Int,
    OC: Int,
    K: Int,
    S: Int,
    P: Int,
    H: Int,
    W: Int,
    BATCH: Int,
](ctx: DeviceContext, label: String) raises -> Bool:
    comptime CB = ConvPCBlock[IC, OC, K, S, P, H, W, PCReLU]
    comptime IN = CB.IN_DIM
    comptime OUT = CB.OUT_DIM
    comptime PSZ = CB.PARAM_SIZE

    # ── Host buffers + CPU reference ──────────────────────────────────────────
    var x_buf = alloc[Scalar[dtype]](BATCH * IN).as_unsafe_any_origin()
    var a_cpu = alloc[Scalar[dtype]](BATCH * IN).as_unsafe_any_origin()
    var mu_cpu = alloc[Scalar[dtype]](BATCH * OUT).as_unsafe_any_origin()
    var eps_buf = alloc[Scalar[dtype]](BATCH * OUT).as_unsafe_any_origin()
    var params_buf = alloc[Scalar[dtype]](PSZ).as_unsafe_any_origin()
    var z_cpu = alloc[Scalar[dtype]](BATCH * IN).as_unsafe_any_origin()
    var grads_cpu = alloc[Scalar[dtype]](PSZ).as_unsafe_any_origin()

    for i in range(BATCH * IN):
        x_buf[i] = Scalar[dtype](sin(Float32(i) * 0.7 + 0.3) * 1.5)
    for i in range(BATCH * OUT):
        eps_buf[i] = Scalar[dtype](sin(Float32(i) * 1.1 + 1.7))
    var params = tile[PSZ](params_buf)
    CB.pc_init_params[PCXavier, dtype](params)

    var x = tile[BATCH, IN](x_buf)
    var a_c = tile[BATCH, IN](a_cpu)
    var mu_c = tile[BATCH, OUT](mu_cpu)
    var eps = tile[BATCH, OUT](eps_buf)
    var z_c = tile[BATCH, IN](z_cpu)
    var g_c = tile[PSZ](grads_cpu)

    CB.predict[BATCH, dtype](x, params, mu_c, a_c)
    CB.pull_back[BATCH, dtype](eps, params, z_c)
    CB.weight_grad[BATCH, dtype](eps, a_c, g_c)

    # ── Device buffers ────────────────────────────────────────────────────────
    var x_d = ctx.enqueue_create_buffer[dtype](BATCH * IN)
    var a_d = ctx.enqueue_create_buffer[dtype](BATCH * IN)
    var mu_d = ctx.enqueue_create_buffer[dtype](BATCH * OUT)
    var eps_d = ctx.enqueue_create_buffer[dtype](BATCH * OUT)
    var params_d = ctx.enqueue_create_buffer[dtype](PSZ)
    var z_d = ctx.enqueue_create_buffer[dtype](BATCH * IN)
    var grads_d = ctx.enqueue_create_buffer[dtype](PSZ)

    # Upload x, eps, params.
    var x_h = ctx.enqueue_create_host_buffer[dtype](BATCH * IN)
    for i in range(BATCH * IN):
        x_h.unsafe_ptr()[i] = x_buf[i]
    ctx.enqueue_copy(x_d, x_h)
    var eps_h = ctx.enqueue_create_host_buffer[dtype](BATCH * OUT)
    for i in range(BATCH * OUT):
        eps_h.unsafe_ptr()[i] = eps_buf[i]
    ctx.enqueue_copy(eps_d, eps_h)
    var p_h = ctx.enqueue_create_host_buffer[dtype](PSZ)
    for i in range(PSZ):
        p_h.unsafe_ptr()[i] = params_buf[i]
    ctx.enqueue_copy(params_d, p_h)

    var x_t = tile[BATCH, IN](x_d)
    var a_t = tile[BATCH, IN](a_d)
    var mu_t = tile[BATCH, OUT](mu_d)
    var eps_t = tile[BATCH, OUT](eps_d)
    var p_t = tile[PSZ](params_d)
    var z_t = tile[BATCH, IN](z_d)
    var g_t = tile[PSZ](grads_d)

    # ── GPU ops ───────────────────────────────────────────────────────────────
    CB.predict_gpu[BATCH, dtype](ctx, x_t, p_t, mu_t, a_t)
    CB.pull_back_gpu[BATCH, dtype](ctx, eps_t, p_t, z_t)
    CB.weight_grad_gpu[BATCH, dtype](ctx, eps_t, a_t, g_t)
    ctx.synchronize()

    # ── Read back ─────────────────────────────────────────────────────────────
    var mu_gpu = ctx.enqueue_create_host_buffer[dtype](BATCH * OUT)
    var a_gpu = ctx.enqueue_create_host_buffer[dtype](BATCH * IN)
    var z_gpu = ctx.enqueue_create_host_buffer[dtype](BATCH * IN)
    var g_gpu = ctx.enqueue_create_host_buffer[dtype](PSZ)
    ctx.enqueue_copy(mu_gpu, mu_d)
    ctx.enqueue_copy(a_gpu, a_d)
    ctx.enqueue_copy(z_gpu, z_d)
    ctx.enqueue_copy(g_gpu, grads_d)
    ctx.synchronize()

    var d_predict = _max_abs_diff(mu_cpu, mu_gpu.unsafe_ptr().as_unsafe_any_origin(), BATCH * OUT)
    var d_abelow = _max_abs_diff(a_cpu, a_gpu.unsafe_ptr().as_unsafe_any_origin(), BATCH * IN)
    var d_pullback = _max_abs_diff(z_cpu, z_gpu.unsafe_ptr().as_unsafe_any_origin(), BATCH * IN)
    var d_wgrad = _max_abs_diff(grads_cpu, g_gpu.unsafe_ptr().as_unsafe_any_origin(), PSZ)

    print("── " + label + " ──")
    print("  predict     max|Δ| =", d_predict)
    print("  a_below     max|Δ| =", d_abelow)
    print("  pull_back   max|Δ| =", d_pullback)
    print("  weight_grad max|Δ| =", d_wgrad)

    var ok = (
        d_predict < TOL
        and d_abelow < TOL
        and d_pullback < TOL
        and d_wgrad < TOL
    )

    x_buf.free()
    a_cpu.free()
    mu_cpu.free()
    eps_buf.free()
    params_buf.free()
    z_cpu.free()
    grads_cpu.free()
    return ok


def main() raises:
    print("ConvPCBlock CPU-vs-GPU parity (P2)\n")
    var ctx = DeviceContext()
    var all_ok = True
    all_ok = run_gpu_parity[2, 3, 3, 1, 1, 4, 4, 2](ctx, "A  s=1 p=1 4x4") and all_ok
    all_ok = run_gpu_parity[2, 3, 3, 2, 1, 5, 5, 2](ctx, "B  s=2 p=1 5x5") and all_ok
    all_ok = run_gpu_parity[1, 4, 3, 1, 0, 6, 6, 3](ctx, "C  s=1 p=0 6x6") and all_ok

    print("")
    if all_ok:
        print("✅ PASS — GPU matches CPU within", TOL)
    else:
        print("❌ FAIL — CPU/GPU parity mismatch")
        raise Error("ConvPCBlock GPU parity failed")
