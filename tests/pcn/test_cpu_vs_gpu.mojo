"""CPU vs GPU parity test for pcn.

Builds a small 3-block PCN, runs `compute_grads_only` on CPU and
`compute_grads_only_gpu` on GPU starting from identical params + identical
input/target, then compares the resulting grads and latents element-wise.

Tolerance:
  - Apple: ≤1e-4 max abs diff (naive kernels do reductions in the same order
    as CPU, so should be near-bitwise; allow some slack for floating-point).
  - NVIDIA: same tolerance for now (we're using naive kernels, no MMA).

Run:
    pixi run -e apple  mojo run -I . tests/pcn/test_cpu_vs_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/pcn/test_cpu_vs_gpu.mojo
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
comptime T_INFER = 5
comptime LR_X: Float64 = 0.05

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

comptime TOL: Float64 = 1.0e-4
comptime SPIKE_SIGMA: Float64 = 0.25   # Σ at the spiking level (α < 1 boosts)


def main() raises:
    print("=" * 60)
    print("pcn CPU vs GPU parity test")
    print("=" * 60)
    print("  arch       : 6 → 8 → 5 → 3")
    print("  PARAM_SIZE :", NET.PARAM_SIZE)
    print("  LATENT_DIM :", NET.LATENT_DIM)
    print("  BATCH=", BATCH, " T_INFER=", T_INFER)
    print("  tolerance  :", TOL)

    var ctx = DeviceContext()

    # ── Allocate params on host, init with Xavier (deterministic seed) ────────
    var params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    memset(params_buf, 0, NET.PARAM_SIZE)
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
    memset(grads_cpu_buf, 0, NET.PARAM_SIZE)
    memset(lat_cpu_buf, 0, BATCH * NET.LATENT_DIM)
    memset(mu_eps_cpu_buf, 0, BATCH * NET.SCRATCH_OUT_DIM)
    memset(a_below_cpu_buf, 0, BATCH * NET.SCRATCH_IN_DIM)
    memset(z_below_cpu_buf, 0, BATCH * NET.SCRATCH_IN_DIM)
    memset(dx_cpu_buf, 0, BATCH * NET.LATENT_DIM)

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

    # Upload params + input/target
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

    # Download grads + latents
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

    # Snapshot the unspiked CPU gradients so the spiked arm below can prove
    # it actually changed something.
    var grads_unspiked_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    for i in range(NET.PARAM_SIZE):
        grads_unspiked_buf[i] = grads_cpu_buf[i]

    print("\n  Parity:")
    print("    max |grads_cpu - grads_gpu|   :", max_grad_diff, " at idx", idx_max_grad)
    print("    max |latents_cpu - latents_gpu|:", max_lat_diff, " at idx", idx_max_lat)
    print("    tolerance                     :", TOL)

    if max_grad_diff <= TOL and max_lat_diff <= TOL:
        print("\n  [PASS] CPU and GPU agree within tolerance")
    else:
        print("\n  [FAIL] CPU vs GPU disagreement exceeds tolerance")
        raise Error("parity test failed")

    # =================================================================
    # Second arm: the SAME comparison with the precision spike ACTIVE.
    #
    # The first arm runs spike_sigma=1, which returns from
    # `_apply_precision_spike_gpu` before enqueueing anything — so it does
    # NOT exercise `_precision_scale_kernel` at all. This arm does.
    # (cf. `_the_package_build_cannot_instantiate_a_generic_kernel`: a green
    # build is not evidence that a GPU kernel runs.)
    # =================================================================
    print("\n  Spiked arm (spike_sigma =", SPIKE_SIGMA, "):")

    var cpu_spiked = TRAINER.compute_grads_only[BATCH](
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
        spike_sigma=Scalar[dtype](SPIKE_SIGMA),
    )
    print("    CPU energy:", cpu_spiked.energy_initial, "→",
          cpu_spiked.energy_final)

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
        spike_sigma=Scalar[dtype](SPIKE_SIGMA),
    )
    ctx.synchronize()
    ctx.enqueue_copy(grads_gpu_host, grads_dbuf)
    ctx.enqueue_copy(lat_gpu_host, lat_dbuf)
    ctx.synchronize()

    var sp_grad_diff: Float64 = 0.0
    var sp_moved: Int = 0
    for i in range(NET.PARAM_SIZE):
        var g_cpu = Float64(grads_cpu_buf[i])
        var g_gpu = Float64(grads_gpu_host.unsafe_ptr()[i])
        var d = mabs(g_cpu - g_gpu)
        if d > sp_grad_diff:
            sp_grad_diff = d
        # Did the spike actually change this gradient vs the unspiked arm?
        if Float64(grads_unspiked_buf[i]) != g_cpu:
            sp_moved += 1

    var sp_lat_diff: Float64 = 0.0
    for i in range(BATCH * NET.LATENT_DIM):
        var d = mabs(
            Float64(lat_cpu_buf[i]) - Float64(lat_gpu_host.unsafe_ptr()[i])
        )
        if d > sp_lat_diff:
            sp_lat_diff = d

    print("    max |grads_cpu - grads_gpu|   :", sp_grad_diff)
    print("    max |latents_cpu - latents_gpu|:", sp_lat_diff)
    print("    spike moved", sp_moved, "of", NET.PARAM_SIZE, "gradients vs the unspiked arm")

    if sp_moved == 0:
        print("\n  [FAIL] spike changed nothing — this arm is vacuous")
        raise Error("spiked parity arm is vacuous")
    if sp_grad_diff <= TOL and sp_lat_diff <= TOL:
        print("\n  [PASS] CPU and GPU agree with the spike active")
    else:
        print("\n  [FAIL] spiked CPU vs GPU disagreement exceeds tolerance")
        raise Error("spiked parity test failed")

    print("=== Done ===")
