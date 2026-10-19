"""MCPC Gaussian toy reproduction — Bogacz notebook 2.

Trains a generative MCPC model on N(1, 5) data, then samples from the
learned distribution and verifies mean + variance match.

Architecture (mirrors notebook 2):
    PCBlock[1, 1, PCIdentity]   # block_0 — the "BiasLayer" / learned prior
    PCBlock[1, 1, PCIdentity]   # block_1 — the "decoder" Linear

With x_in held at 1.0, block_0 emits μ_0 = W_0 + b_0 (a learned constant).
The interior latent x_1 settles around μ_0 via SGLD; samples are produced
by forwarding x_1 through block_1.

Hyperparameters (notebook 2 defaults):
    T_mixing = 250, T_sampling = 1
    lr_x = 0.01, noise_var = 1.0
    Adam lr = 0.1
    batch = 256, 150 batches/epoch × 5 epochs
    Generation: T = 2000 iterations, 1024 samples

Pass criterion (loose, since this is a stochastic toy):
    |sample_mean - 1.0| ≤ 0.5
    |sample_var  - 5.0| ≤ 2.5

Run:
    pixi run mojo run -I . tests/pcn/test_mcpc_gaussian.mojo
"""

from std.math import sqrt, log, cos, pi
from std.memory import alloc, memset
from std.random.philox import Random as PhiloxRandom
from noeira.core.tiles import tile

from noeira.nn.constants import DT as dtype
from noeira.experimental.pcn.pc_initializer import PCXavier
from noeira.experimental.pcn.pc_optimizer import PCAdam
from noeira.experimental.pcn import (
    PCBlock,
    PCSequential,
    PCIdentity,
    PCTrainer,
)


comptime BATCH = 256
comptime EPOCHS = 5
comptime N_BATCHES_PER_EPOCH = 150
comptime T_MIXING = 250
comptime T_SAMPLING = 1
comptime LR_X: Float64 = 0.01
comptime NOISE_VAR: Float64 = 1.0
comptime ADAM_LR: Float64 = 0.1

comptime DATA_MEAN: Float64 = 1.0
comptime DATA_VAR: Float64 = 5.0

comptime GEN_BATCH = 1024
comptime GEN_T = 2000

comptime NET = PCSequential[
    PCBlock[1, 1, PCIdentity],   # prior (input=1 → 1-d latent above)
    PCBlock[1, 1, PCIdentity],   # decoder (1-d latent → 1-d output)
]
comptime TRAINER = PCTrainer[
    PCBlock[1, 1, PCIdentity],
    PCBlock[1, 1, PCIdentity],
    dtype=dtype,
]
comptime OPT = PCAdam[LR=ADAM_LR]


def main() raises:
    print("=" * 60)
    print("MCPC Gaussian toy — Bogacz notebook 2 reproduction")
    print("=" * 60)
    print("  arch       : PCPrior[1,1] → x_1 → Linear[1,1] → output ↔ data")
    print("  PARAM_SIZE :", NET.PARAM_SIZE)
    print("  LATENT_DIM :", NET.LATENT_DIM)
    print("  data       : N(", DATA_MEAN, ",", DATA_VAR, ")")
    print("  hyperparams: BATCH=", BATCH, " T_MIX=", T_MIXING, " EPOCHS=", EPOCHS)
    print("  SGLD       : lr_x=", LR_X, " noise_var=", NOISE_VAR)
    print("  Adam lr    :", ADAM_LR)

    # ── params + grads + Adam state (CPU) ────────────────────────────────────
    var params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var grads_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var opt_state_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE * OPT.STATE_PER_PARAM).as_unsafe_any_origin()
    var opt_global_buf = alloc[Scalar[dtype]](OPT.GLOBAL_STATE_SIZE).as_unsafe_any_origin()
    memset(params_buf, 0, NET.PARAM_SIZE)
    memset(grads_buf, 0, NET.PARAM_SIZE)
    memset(opt_state_buf, 0, NET.PARAM_SIZE * OPT.STATE_PER_PARAM)
    memset(opt_global_buf, 0, OPT.GLOBAL_STATE_SIZE)

    var params = tile[NET.PARAM_SIZE](params_buf)
    var grads = tile[NET.PARAM_SIZE](grads_buf)
    var opt_state = tile[NET.PARAM_SIZE, OPT.STATE_PER_PARAM](opt_state_buf)
    var opt_global = tile[OPT.GLOBAL_STATE_SIZE](opt_global_buf)
    NET.pc_init_params[PCXavier, dtype](params)

    # ── scratch buffers ──────────────────────────────────────────────────────
    var lat_buf = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var mu_eps_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_OUT_DIM).as_unsafe_any_origin()
    var a_below_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var z_below_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var dx_buf_raw = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var noise_buf_raw = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    memset(lat_buf, 0, BATCH * NET.LATENT_DIM)
    memset(mu_eps_buf_raw, 0, BATCH * NET.SCRATCH_OUT_DIM)
    memset(a_below_buf_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
    memset(z_below_buf_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
    memset(dx_buf_raw, 0, BATCH * NET.LATENT_DIM)
    memset(noise_buf_raw, 0, BATCH * NET.LATENT_DIM)

    var latents = tile[BATCH, NET.LATENT_DIM](lat_buf)
    var mu_eps_buf = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_eps_buf_raw)
    var a_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](a_below_buf_raw)
    var z_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](z_below_buf_raw)
    var dx_buf = tile[BATCH, NET.LATENT_DIM](dx_buf_raw)
    var noise_buf = tile[BATCH, NET.LATENT_DIM](noise_buf_raw)

    # ── x_in is always 1.0 (pseudo-input — drives the bias-only block_0) ─────
    var x_in_buf = alloc[Scalar[dtype]](BATCH * NET.IN_DIM).as_unsafe_any_origin()
    for i in range(BATCH * NET.IN_DIM):
        x_in_buf[i] = Scalar[dtype](1.0)
    var x_in = tile[BATCH, NET.IN_DIM](x_in_buf)

    var y_data_buf = alloc[Scalar[dtype]](BATCH * NET.OUT_DIM).as_unsafe_any_origin()
    var y_data = tile[BATCH, NET.OUT_DIM](y_data_buf)

    # ── Train ────────────────────────────────────────────────────────────────
    print("\n  epoch | last_loss")
    print("  ------+----------")

    var step_num: Int = 0
    var noise_offset: UInt64 = 1000
    var data_rng = PhiloxRandom(seed=UInt64(7), offset=UInt64(0))

    for epoch in range(EPOCHS):
        var last_loss: Float64 = 0.0
        for batch_idx in range(N_BATCHES_PER_EPOCH):
            # Generate fresh data: y ~ N(DATA_MEAN, DATA_VAR)
            for i in range(BATCH * NET.OUT_DIM):
                # Box-Muller: 2 uniforms → 1 normal (we discard the second)
                var u1 = data_rng.step_uniform()[0]
                var u2 = data_rng.step_uniform()[0]
                if u1 < 1e-10:
                    u1 = 1e-10
                var z = sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)
                y_data_buf[i] = Scalar[dtype](
                    DATA_MEAN + sqrt(DATA_VAR) * Float64(z)
                )

            var result = TRAINER.compute_grads_only_mcpc[BATCH](
                params, grads, latents,
                mu_eps_buf, a_below_buf, z_below_buf, dx_buf, noise_buf,
                x_in, y_data,
                T_mixing=T_MIXING,
                T_sampling=T_SAMPLING,
                lr_x=Scalar[dtype](LR_X),
                noise_var=Scalar[dtype](NOISE_VAR),
                seed=UInt64(42),
                offset_base=noise_offset,
            )
            # Bump noise offset enough to cover the entire SGLD trajectory
            noise_offset += UInt64(BATCH * NET.LATENT_DIM * (T_MIXING + T_SAMPLING) * 2)

            step_num += 1
            OPT.step[NET.PARAM_SIZE, dtype](
                params, grads, opt_state, opt_global, step_num
            )
            last_loss = result.output_loss_final

        print("    ", epoch, "  ", fit(String(last_loss), 9))

    # ── Inspect trained params ───────────────────────────────────────────────
    print("\n  Trained params:")
    print("    block_0 W :", Float64(params_buf[0]))
    print("    block_0 b :", Float64(params_buf[1]))
    print("    block_1 W :", Float64(params_buf[2]))
    print("    block_1 b :", Float64(params_buf[3]))
    var prior_mean = Float64(params_buf[0]) + Float64(params_buf[1])
    var decoder_W = Float64(params_buf[2])
    var decoder_b = Float64(params_buf[3])
    print("    → prior mean (W_0 + b_0) :", prior_mean)
    print("    → expected output mean   :", decoder_W * prior_mean + decoder_b)

    # ── Generate samples (reuse training scratch — but with GEN_BATCH=1024
    #     we need bigger buffers, so allocate fresh). ─────────────────────────
    var gen_lat_buf = alloc[Scalar[dtype]](GEN_BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var gen_mu_eps_buf_raw = alloc[Scalar[dtype]](GEN_BATCH * NET.SCRATCH_OUT_DIM).as_unsafe_any_origin()
    var gen_a_below_buf_raw = alloc[Scalar[dtype]](GEN_BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var gen_z_below_buf_raw = alloc[Scalar[dtype]](GEN_BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var gen_dx_buf_raw = alloc[Scalar[dtype]](GEN_BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var gen_noise_buf_raw = alloc[Scalar[dtype]](GEN_BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var gen_sample_buf = alloc[Scalar[dtype]](GEN_BATCH * NET.OUT_DIM).as_unsafe_any_origin()
    var gen_x_in_buf = alloc[Scalar[dtype]](GEN_BATCH * NET.IN_DIM).as_unsafe_any_origin()
    var gen_y_dummy_buf = alloc[Scalar[dtype]](GEN_BATCH * NET.OUT_DIM).as_unsafe_any_origin()
    memset(gen_lat_buf, 0, GEN_BATCH * NET.LATENT_DIM)
    memset(gen_mu_eps_buf_raw, 0, GEN_BATCH * NET.SCRATCH_OUT_DIM)
    memset(gen_a_below_buf_raw, 0, GEN_BATCH * NET.SCRATCH_IN_DIM)
    memset(gen_z_below_buf_raw, 0, GEN_BATCH * NET.SCRATCH_IN_DIM)
    memset(gen_dx_buf_raw, 0, GEN_BATCH * NET.LATENT_DIM)
    memset(gen_noise_buf_raw, 0, GEN_BATCH * NET.LATENT_DIM)
    memset(gen_sample_buf, 0, GEN_BATCH * NET.OUT_DIM)
    memset(gen_y_dummy_buf, 0, GEN_BATCH * NET.OUT_DIM)
    for i in range(GEN_BATCH * NET.IN_DIM):
        gen_x_in_buf[i] = Scalar[dtype](1.0)

    var gen_lat = tile[GEN_BATCH, NET.LATENT_DIM](gen_lat_buf)
    var gen_mu_eps = tile[GEN_BATCH, NET.SCRATCH_OUT_DIM](gen_mu_eps_buf_raw)
    var gen_a_below = tile[GEN_BATCH, NET.SCRATCH_IN_DIM](gen_a_below_buf_raw)
    var gen_z_below = tile[GEN_BATCH, NET.SCRATCH_IN_DIM](gen_z_below_buf_raw)
    var gen_dx = tile[GEN_BATCH, NET.LATENT_DIM](gen_dx_buf_raw)
    var gen_noise = tile[GEN_BATCH, NET.LATENT_DIM](gen_noise_buf_raw)
    var gen_x_in = tile[GEN_BATCH, NET.IN_DIM](gen_x_in_buf)
    var gen_y_dummy = tile[GEN_BATCH, NET.OUT_DIM](gen_y_dummy_buf)
    var gen_sample = tile[GEN_BATCH, NET.OUT_DIM](gen_sample_buf)

    # ── Sanity-check Box-Muller noise: should be ~N(0, 1) ───────────────────
    var rng_test = PhiloxRandom(seed=UInt64(7), offset=UInt64(2_000_000))
    var n_total: Float64 = 0
    var nsq_total: Float64 = 0
    var n_count = 4096
    for k in range(n_count):
        var u1 = Float64(rng_test.step_uniform()[0])
        var u2 = Float64(rng_test.step_uniform()[0])
        if u1 < 1e-10:
            u1 = 1e-10
        var r = sqrt(-2.0 * log(u1))
        var z = r * cos(2.0 * pi * u2)
        n_total += z
        nsq_total += z * z
    var n_mean = n_total / Float64(n_count)
    var n_var = nsq_total / Float64(n_count) - n_mean * n_mean
    print("\n  Box-Muller reference: mean=", n_mean, " var=", n_var)

    print("\n  [generation]: T=", GEN_T, ", N=", GEN_BATCH)
    TRAINER.generate_samples[GEN_BATCH](
        params, gen_lat, gen_mu_eps, gen_a_below, gen_z_below, gen_dx, gen_noise,
        gen_x_in, gen_y_dummy, gen_sample,
        T=GEN_T,
        lr_x=Scalar[dtype](LR_X),
        noise_var=Scalar[dtype](NOISE_VAR),
        seed=UInt64(99),
        offset_base=UInt64(0),
    )

    # ── Diagnostic: latent stats at end of generation ────────────────────────
    var lat_sum: Float64 = 0
    for i in range(GEN_BATCH):
        lat_sum += Float64(gen_lat_buf[i])
    var lat_mean = lat_sum / Float64(GEN_BATCH)
    var lat_sum_sq: Float64 = 0
    for i in range(GEN_BATCH):
        var d = Float64(gen_lat_buf[i]) - lat_mean
        lat_sum_sq += d * d
    var lat_var = lat_sum_sq / Float64(GEN_BATCH)
    print("  Latent x_1 at end-of-gen: mean=", lat_mean, " var=", lat_var)

    # ── Compute sample stats ─────────────────────────────────────────────────
    var sum_v: Float64 = 0
    for i in range(GEN_BATCH):
        sum_v += Float64(gen_sample_buf[i])
    var sample_mean = sum_v / Float64(GEN_BATCH)

    var sum_sq: Float64 = 0
    for i in range(GEN_BATCH):
        var d = Float64(gen_sample_buf[i]) - sample_mean
        sum_sq += d * d
    var sample_var = sum_sq / Float64(GEN_BATCH)

    print("\n  Target mean :", DATA_MEAN, "    sample mean :", sample_mean)
    print("  Target var  :", DATA_VAR, "    sample var  :", sample_var)
    print("  |Δmean| =", sample_mean - DATA_MEAN if sample_mean > DATA_MEAN else DATA_MEAN - sample_mean)
    print("  |Δvar|  =", sample_var - DATA_VAR if sample_var > DATA_VAR else DATA_VAR - sample_var)

    var dmean = sample_mean - DATA_MEAN if sample_mean > DATA_MEAN else DATA_MEAN - sample_mean
    var dvar = sample_var - DATA_VAR if sample_var > DATA_VAR else DATA_VAR - sample_var
    if dmean <= 0.5 and dvar <= 2.5:
        print("\n  [PASS] MCPC learned the N(1, 5) data distribution")
    else:
        print("\n  [FAIL] sample stats out of tolerance")
        raise Error("MCPC test failed")

    # cleanup
    params_buf.free()
    grads_buf.free()
    opt_state_buf.free()
    opt_global_buf.free()
    lat_buf.free()
    mu_eps_buf_raw.free()
    a_below_buf_raw.free()
    z_below_buf_raw.free()
    dx_buf_raw.free()
    noise_buf_raw.free()
    x_in_buf.free()
    y_data_buf.free()
    gen_lat_buf.free()
    gen_mu_eps_buf_raw.free()
    gen_a_below_buf_raw.free()
    gen_z_below_buf_raw.free()
    gen_dx_buf_raw.free()
    gen_noise_buf_raw.free()
    gen_sample_buf.free()
    gen_x_in_buf.free()
    gen_y_dummy_buf.free()
    print("=== Done ===")

from noeira.core.fmt import fit