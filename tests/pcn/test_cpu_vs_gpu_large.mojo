"""CPU vs GPU parity test for pcn at realistic dimensions.

Stresses the optimized GPU matmul path (`linalg.matmul` on NVIDIA, 2×2 tile
on Apple) with a 4-block PCN sized to actually exercise the GEMM tiling:

    BATCH = 64, arch: 64 → 256 → 256 → 128 → 10

NVIDIA's `linalg.matmul` reduces along K in tile-blocked order, which diverges
from CPU's row-major sum — float32 addition is non-associative. Over
T_INFER=4 inference steps with K=256, the absolute differences observed in
practice are ~1.6e-2 (max) on grads, which is ~5e-4 relative to the
output-scale magnitudes. We assert a *relative* tolerance (1%) instead of
absolute so the test scales sanely with workload size; absolute diffs are
still printed for diagnostics.

Apple's 2×2 tile happens to reduce in the same order as the CPU nested
loops, so on Apple the diff is typically 0.0 (bitwise).

Run:
    pixi run -e apple  mojo run -I . tests/pcn/test_cpu_vs_gpu_large.mojo
    pixi run -e nvidia mojo run -I . tests/pcn/test_cpu_vs_gpu_large.mojo
"""

from std.math import abs as mabs
from std.memory import alloc, unsafe_memset
from std.random.philox import Random as PhiloxRandom
from max.gpu.host import DeviceContext
from noeira.core.tiles import tile

from noeira.nn.constants import DT as dtype
from noeira.experimental.pcn.pc_initializer import PCXavier
from noeira.experimental.pcn import (
    PCBlock,
    PCSequential,
    PCIdentity,
    PCReLU,
    PCTrainer,
)


comptime BATCH = 64
comptime T_INFER = 4
comptime LR_X: Float64 = 0.05

comptime NET = PCSequential[
    PCBlock[64, 256, PCIdentity],
    PCBlock[256, 256, PCReLU],
    PCBlock[256, 128, PCReLU],
    PCBlock[128, 10, PCReLU],
]
comptime TRAINER = PCTrainer[
    PCBlock[64, 256, PCIdentity],
    PCBlock[256, 256, PCReLU],
    PCBlock[256, 128, PCReLU],
    PCBlock[128, 10, PCReLU],
    dtype=dtype,
]

# Relative tolerance: max |cpu - gpu| / max |cpu| must stay under this.
# 1% is comfortable for a 4-block PCN with K=256 and T_INFER=4 in float32;
# real algorithmic bugs blow far past this.
comptime REL_TOL: Float64 = 1.0e-2


def main() raises:
    print("=" * 60)
    print("pcn CPU vs GPU parity test (large dims)")
    print("=" * 60)
    print("  arch       : 64 → 256 → 256 → 128 → 10")
    print("  PARAM_SIZE :", NET.PARAM_SIZE)
    print("  LATENT_DIM :", NET.LATENT_DIM)
    print("  BATCH=", BATCH, " T_INFER=", T_INFER)
    print("  rel tolerance:", REL_TOL)

    var ctx = DeviceContext()

    # ── Allocate params on host, init with Xavier (deterministic seed) ────────
    var params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    unsafe_memset(params_buf, 0, NET.PARAM_SIZE)
    var params = tile[NET.PARAM_SIZE](params_buf)
    NET.pc_init_params[PCXavier, dtype](params)

    # ── Allocate input + target (deterministic Philox) ────────────────────────
    var x_in_buf = alloc[Scalar[dtype]](BATCH * NET.IN_DIM).as_unsafe_any_origin()
    var y_tgt_buf = alloc[Scalar[dtype]](BATCH * NET.OUT_DIM).as_unsafe_any_origin()
    var rng = PhiloxRandom(seed=UInt64(13), offset=UInt64(0))
    for i in range(BATCH * NET.IN_DIM):
        var r = rng.step_uniform()
        x_in_buf[i] = Scalar[dtype](Float32(r[0]) * 2.0 - 1.0)
    for i in range(BATCH * NET.OUT_DIM):
        y_tgt_buf[i] = Scalar[dtype](0)
    for b in range(BATCH):
        y_tgt_buf[b * NET.OUT_DIM + (b % NET.OUT_DIM)] = Scalar[dtype](1.0)

    var x_in = tile[BATCH, NET.IN_DIM](x_in_buf)
    var y_target = tile[BATCH, NET.OUT_DIM](y_tgt_buf)

    # =================================================================
    # CPU run
    # =================================================================
    var grads_cpu_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var lat_cpu_buf = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var mu_eps_cpu_buf = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_OUT_DIM).as_unsafe_any_origin()
    var a_below_cpu_buf = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var z_below_cpu_buf = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var dx_cpu_buf = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    unsafe_memset(grads_cpu_buf, 0, NET.PARAM_SIZE)
    unsafe_memset(lat_cpu_buf, 0, BATCH * NET.LATENT_DIM)
    unsafe_memset(mu_eps_cpu_buf, 0, BATCH * NET.SCRATCH_OUT_DIM)
    unsafe_memset(a_below_cpu_buf, 0, BATCH * NET.SCRATCH_IN_DIM)
    unsafe_memset(z_below_cpu_buf, 0, BATCH * NET.SCRATCH_IN_DIM)
    unsafe_memset(dx_cpu_buf, 0, BATCH * NET.LATENT_DIM)

    var grads_cpu = tile[NET.PARAM_SIZE](grads_cpu_buf)
    var lat_cpu = tile[BATCH, NET.LATENT_DIM](lat_cpu_buf)
    var mu_eps_cpu = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_eps_cpu_buf)
    var a_below_cpu = tile[BATCH, NET.SCRATCH_IN_DIM](a_below_cpu_buf)
    var z_below_cpu = tile[BATCH, NET.SCRATCH_IN_DIM](z_below_cpu_buf)
    var dx_cpu = tile[BATCH, NET.LATENT_DIM](dx_cpu_buf)

    var cpu_result = TRAINER.compute_grads_only[BATCH](
        params,
        grads_cpu,
        lat_cpu,
        mu_eps_cpu,
        a_below_cpu,
        z_below_cpu,
        dx_cpu,
        x_in,
        y_target,
        T_infer=T_INFER,
        lr_x=Scalar[dtype](LR_X),
    )

    print("\n  CPU run:")
    print("    energy_initial:", cpu_result.energy_initial)
    print("    energy_final  :", cpu_result.energy_final)
    print("    output_loss   :", cpu_result.output_loss_final)

    # =================================================================
    # GPU run — same params, same input/target
    # =================================================================
    var params_dbuf = ctx.enqueue_create_buffer[dtype](NET.PARAM_SIZE)
    var grads_dbuf = ctx.enqueue_create_buffer[dtype](NET.PARAM_SIZE)
    var lat_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.LATENT_DIM)
    var mu_eps_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.SCRATCH_OUT_DIM)
    var a_below_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.SCRATCH_IN_DIM)
    var z_below_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.SCRATCH_IN_DIM)
    var dx_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.LATENT_DIM)
    var x_in_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.IN_DIM)
    var y_target_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.OUT_DIM)

    var params_host = ctx.enqueue_create_host_buffer[dtype](NET.PARAM_SIZE)
    for i in range(NET.PARAM_SIZE):
        params_host.unsafe_ptr()[i] = params_buf[i]
    ctx.enqueue_copy(params_dbuf, params_host)

    var x_host = ctx.enqueue_create_host_buffer[dtype](BATCH * NET.IN_DIM)
    for i in range(BATCH * NET.IN_DIM):
        x_host.unsafe_ptr()[i] = x_in_buf[i]
    ctx.enqueue_copy(x_in_dbuf, x_host)

    var y_host = ctx.enqueue_create_host_buffer[dtype](BATCH * NET.OUT_DIM)
    for i in range(BATCH * NET.OUT_DIM):
        y_host.unsafe_ptr()[i] = y_tgt_buf[i]
    ctx.enqueue_copy(y_target_dbuf, y_host)

    var params_t_gpu = tile[NET.PARAM_SIZE](params_dbuf)
    var grads_t_gpu = tile[NET.PARAM_SIZE](grads_dbuf)
    var lat_t_gpu = tile[BATCH, NET.LATENT_DIM](lat_dbuf)
    var mu_eps_t_gpu = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_eps_dbuf)
    var a_below_t_gpu = tile[BATCH, NET.SCRATCH_IN_DIM](a_below_dbuf)
    var z_below_t_gpu = tile[BATCH, NET.SCRATCH_IN_DIM](z_below_dbuf)
    var dx_t_gpu = tile[BATCH, NET.LATENT_DIM](dx_dbuf)
    var x_in_t_gpu = tile[BATCH, NET.IN_DIM](x_in_dbuf)
    var y_target_t_gpu = tile[BATCH, NET.OUT_DIM](y_target_dbuf)

    TRAINER.compute_grads_only_gpu[BATCH](
        ctx,
        params_t_gpu,
        grads_t_gpu,
        lat_t_gpu,
        mu_eps_t_gpu,
        a_below_t_gpu,
        z_below_t_gpu,
        dx_t_gpu,
        x_in_t_gpu,
        y_target_t_gpu,
        T_infer=T_INFER,
        lr_x=Scalar[dtype](LR_X),
    )
    ctx.synchronize()

    var grads_gpu_host = ctx.enqueue_create_host_buffer[dtype](NET.PARAM_SIZE)
    ctx.enqueue_copy(grads_gpu_host, grads_dbuf)
    var lat_gpu_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * NET.LATENT_DIM
    )
    ctx.enqueue_copy(lat_gpu_host, lat_dbuf)
    ctx.synchronize()

    # =================================================================
    # Compare (relative tolerance: max abs diff / max |cpu|)
    # =================================================================
    var max_grad_diff: Float64 = 0.0
    var max_grad_mag: Float64 = 0.0
    var idx_max_grad: Int = 0
    for i in range(NET.PARAM_SIZE):
        var g_cpu = Float64(grads_cpu_buf[i])
        var g_gpu = Float64(grads_gpu_host.unsafe_ptr()[i])
        var d = mabs(g_cpu - g_gpu)
        var m = mabs(g_cpu)
        if d > max_grad_diff:
            max_grad_diff = d
            idx_max_grad = i
        if m > max_grad_mag:
            max_grad_mag = m

    var max_lat_diff: Float64 = 0.0
    var max_lat_mag: Float64 = 0.0
    var idx_max_lat: Int = 0
    for i in range(BATCH * NET.LATENT_DIM):
        var l_cpu = Float64(lat_cpu_buf[i])
        var l_gpu = Float64(lat_gpu_host.unsafe_ptr()[i])
        var d = mabs(l_cpu - l_gpu)
        var m = mabs(l_cpu)
        if d > max_lat_diff:
            max_lat_diff = d
            idx_max_lat = i
        if m > max_lat_mag:
            max_lat_mag = m

    var grad_denom = max_grad_mag if max_grad_mag > 1.0e-6 else 1.0
    var lat_denom = max_lat_mag if max_lat_mag > 1.0e-6 else 1.0
    var rel_grad = max_grad_diff / grad_denom
    var rel_lat = max_lat_diff / lat_denom

    print("\n  Parity:")
    print("    max |grads_cpu - grads_gpu|   :", max_grad_diff, " at idx", idx_max_grad)
    print("      (max |grads_cpu|             :", max_grad_mag, ")")
    print("      (relative                    :", rel_grad, ")")
    print("    max |latents_cpu - latents_gpu|:", max_lat_diff, " at idx", idx_max_lat)
    print("      (max |latents_cpu|           :", max_lat_mag, ")")
    print("      (relative                    :", rel_lat, ")")
    print("    rel tolerance                  :", REL_TOL)

    if rel_grad <= REL_TOL and rel_lat <= REL_TOL:
        print("\n  [PASS] CPU and GPU agree within relative tolerance")
    else:
        print("\n  [FAIL] CPU vs GPU disagreement exceeds relative tolerance")
        raise Error("parity test failed")

    params_buf.free()
    x_in_buf.free()
    y_tgt_buf.free()
    grads_cpu_buf.free()
    lat_cpu_buf.free()
    mu_eps_cpu_buf.free()
    a_below_cpu_buf.free()
    z_below_cpu_buf.free()
    dx_cpu_buf.free()
    print("=== Done ===")
