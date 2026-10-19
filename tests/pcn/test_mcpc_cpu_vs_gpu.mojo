"""CPU vs GPU MCPC parity test for pcn.

Same shape as `test_cpu_vs_gpu.mojo` but exercises the SGLD-aware MCPC path
(`compute_grads_only_mcpc` / `compute_grads_only_mcpc_gpu`). Both calls are
fed the same Philox seed + offset_base so the noise sequence is identical
modulo dtype:

  - CPU `_box_muller_fill` does the Box-Muller transform in Float64.
  - GPU `_box_muller_fill_kernel` does it in Float32.

The ULP-scale dtype difference in the noise feeds back through every SGLD
step, so we use a slacker tolerance (1e-3) than the deterministic test (1e-4).

Run:
    pixi run -e apple  mojo run -I . tests/pcn/test_mcpc_cpu_vs_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/pcn/test_mcpc_cpu_vs_gpu.mojo
"""

from std.math import abs as mabs
from std.memory import alloc, memset
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


comptime BATCH = 4
comptime T_MIXING = 4
comptime T_SAMPLING = 1
comptime LR_X: Float64 = 0.05
comptime NOISE_VAR: Float64 = 0.01
comptime SEED: UInt64 = UInt64(424242)
comptime OFFSET_BASE: UInt64 = UInt64(0)

comptime NET = PCSequential[
    PCBlock[6, 8, PCIdentity],
    PCBlock[8, 5, PCReLU],
    PCBlock[5, 3, PCReLU],
]
comptime TRAINER = PCTrainer[
    PCBlock[6, 8, PCIdentity],
    PCBlock[8, 5, PCReLU],
    PCBlock[5, 3, PCReLU],
    dtype=dtype,
]

# CPU box-muller is Float64, GPU is Float32 — the per-step noise differs at
# ULP scale, which compounds through (T_MIXING + T_SAMPLING) SGLD steps.
# 1e-3 is comfortable; tighter would risk false negatives on tiny dims.
comptime TOL: Float64 = 1.0e-3


def main() raises:
    print("=" * 60)
    print("pcn MCPC CPU vs GPU parity test")
    print("=" * 60)
    print("  arch       : 6 → 8 → 5 → 3")
    print("  PARAM_SIZE :", NET.PARAM_SIZE)
    print("  LATENT_DIM :", NET.LATENT_DIM)
    print("  BATCH=", BATCH, " T_MIXING=", T_MIXING, " T_SAMPLING=", T_SAMPLING)
    print("  noise_var  :", NOISE_VAR)
    print("  seed       :", SEED, " offset_base:", OFFSET_BASE)
    print("  tolerance  :", TOL)

    var ctx = DeviceContext()

    # ── Allocate params on host, init with Xavier (deterministic seed) ────────
    var params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    memset(params_buf, 0, NET.PARAM_SIZE)
    var params = tile[NET.PARAM_SIZE](params_buf)
    NET.pc_init_params[PCXavier, dtype](params)

    # ── Input + target (deterministic Philox) ─────────────────────────────────
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
    var noise_cpu_buf = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    memset(grads_cpu_buf, 0, NET.PARAM_SIZE)
    memset(lat_cpu_buf, 0, BATCH * NET.LATENT_DIM)
    memset(mu_eps_cpu_buf, 0, BATCH * NET.SCRATCH_OUT_DIM)
    memset(a_below_cpu_buf, 0, BATCH * NET.SCRATCH_IN_DIM)
    memset(z_below_cpu_buf, 0, BATCH * NET.SCRATCH_IN_DIM)
    memset(dx_cpu_buf, 0, BATCH * NET.LATENT_DIM)
    memset(noise_cpu_buf, 0, BATCH * NET.LATENT_DIM)

    var grads_cpu = tile[NET.PARAM_SIZE](grads_cpu_buf)
    var lat_cpu = tile[BATCH, NET.LATENT_DIM](lat_cpu_buf)
    var mu_eps_cpu = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_eps_cpu_buf)
    var a_below_cpu = tile[BATCH, NET.SCRATCH_IN_DIM](a_below_cpu_buf)
    var z_below_cpu = tile[BATCH, NET.SCRATCH_IN_DIM](z_below_cpu_buf)
    var dx_cpu = tile[BATCH, NET.LATENT_DIM](dx_cpu_buf)
    var noise_cpu = tile[BATCH, NET.LATENT_DIM](noise_cpu_buf)

    var cpu_result = TRAINER.compute_grads_only_mcpc[BATCH](
        params,
        grads_cpu,
        lat_cpu,
        mu_eps_cpu,
        a_below_cpu,
        z_below_cpu,
        dx_cpu,
        noise_cpu,
        x_in,
        y_target,
        T_mixing=T_MIXING,
        T_sampling=T_SAMPLING,
        lr_x=Scalar[dtype](LR_X),
        noise_var=Scalar[dtype](NOISE_VAR),
        seed=SEED,
        offset_base=OFFSET_BASE,
    )

    print("\n  CPU run:")
    print("    energy_final  :", cpu_result.energy_final)
    print("    output_loss   :", cpu_result.output_loss_final)

    # =================================================================
    # GPU run — same params, same input/target, same seed/offset_base
    # =================================================================
    var params_dbuf = ctx.enqueue_create_buffer[dtype](NET.PARAM_SIZE)
    var grads_dbuf = ctx.enqueue_create_buffer[dtype](NET.PARAM_SIZE)
    var lat_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.LATENT_DIM)
    var mu_eps_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.SCRATCH_OUT_DIM)
    var a_below_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.SCRATCH_IN_DIM)
    var z_below_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.SCRATCH_IN_DIM)
    var dx_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.LATENT_DIM)
    var noise_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.LATENT_DIM)
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
    var noise_t_gpu = tile[BATCH, NET.LATENT_DIM](noise_dbuf)
    var x_in_t_gpu = tile[BATCH, NET.IN_DIM](x_in_dbuf)
    var y_target_t_gpu = tile[BATCH, NET.OUT_DIM](y_target_dbuf)

    TRAINER.compute_grads_only_mcpc_gpu[BATCH](
        ctx,
        params_t_gpu,
        grads_t_gpu,
        lat_t_gpu,
        mu_eps_t_gpu,
        a_below_t_gpu,
        z_below_t_gpu,
        dx_t_gpu,
        noise_t_gpu,
        x_in_t_gpu,
        y_target_t_gpu,
        T_mixing=T_MIXING,
        T_sampling=T_SAMPLING,
        lr_x=Scalar[dtype](LR_X),
        noise_var=Scalar[dtype](NOISE_VAR),
        seed=SEED,
        offset_base=OFFSET_BASE,
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
    # Compare
    # =================================================================
    var max_grad_diff: Float64 = 0.0
    var idx_max_grad: Int = 0
    for i in range(NET.PARAM_SIZE):
        var g_cpu = Float64(grads_cpu_buf[i])
        var g_gpu = Float64(grads_gpu_host.unsafe_ptr()[i])
        var d = mabs(g_cpu - g_gpu)
        if d > max_grad_diff:
            max_grad_diff = d
            idx_max_grad = i

    var max_lat_diff: Float64 = 0.0
    var idx_max_lat: Int = 0
    for i in range(BATCH * NET.LATENT_DIM):
        var l_cpu = Float64(lat_cpu_buf[i])
        var l_gpu = Float64(lat_gpu_host.unsafe_ptr()[i])
        var d = mabs(l_cpu - l_gpu)
        if d > max_lat_diff:
            max_lat_diff = d
            idx_max_lat = i

    print("\n  Parity:")
    print("    max |grads_cpu - grads_gpu|   :", max_grad_diff, " at idx", idx_max_grad)
    print("    max |latents_cpu - latents_gpu|:", max_lat_diff, " at idx", idx_max_lat)
    print("    tolerance                     :", TOL)

    if max_grad_diff <= TOL and max_lat_diff <= TOL:
        print("\n  [PASS] CPU and GPU agree within tolerance")
    else:
        print("\n  [FAIL] CPU vs GPU disagreement exceeds tolerance")
        raise Error("MCPC parity test failed")

    print("=== Done ===")
