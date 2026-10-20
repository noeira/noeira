"""Stochastic tPC — Step 1 of PCN_WORLD_MODEL_ROADMAP.md.

Combines MCPC's SGLD posterior inference with tPC's recurrent dynamics:
    μ_z_t = W_r·tanh(z_{t-1}) + b
    z_t  ←  z_t − lr_x·∂E/∂z_t + sqrt(2·noise_var·lr_x)·N(0,1)
    y_t  =  W_dec·tanh(z_t) + b   ↔   data[t] + ε_obs   (clamped during training)

This is the canonical iterative version of the latent-state model used by
Dreamer/PlaNet/TD-MPC2 (their amortized encoders replace the SGLD chain).

Two phases run back-to-back in a single test, reusing all buffers but with
fresh params + Adam state:
  Phase A (stochastic): obs corrupted with σ_obs=0.1, SGLD with noise_var>0,
                       recall on CLEAN data must beat zero baseline by ≥30%.
  Phase B (degenerate): clean obs, noise_var=0 → reduces to plain tPC. Same
                       recall criterion must hold (sanity check that the MCPC
                       path is a strict superset of the deterministic path).

Run:
    pixi run mojo run -I . tests/pcn/test_stochastic_tpc.mojo
"""

from std.math import sqrt, log, cos, pi
from std.memory import alloc, unsafe_memset
from std.random.philox import Random as PhiloxRandom
from std.time import perf_counter_ns
from noeira.core.tiles import tile

from noeira.nn.constants import DT as dtype
from noeira.experimental.pcn.pc_initializer import PCXavier
from noeira.experimental.pcn.pc_optimizer import PCAdam
from noeira.nn.datasets.mnist import MNIST
from noeira.experimental.pcn import (
    PCBlock,
    PCSequential,
    PCTanh,
    PCTrainer,
)


comptime BATCH = 1
comptime HIDDEN = 64
comptime DATA_DIM = 784
comptime SEQ_LEN = 5
comptime EPOCHS = 100
comptime T_MIXING = 50
comptime T_SAMPLING = 1
comptime LR_X: Float64 = 0.01
comptime ADAM_LR: Float64 = 0.005

# Phase A — stochastic
comptime OBS_NOISE_STD: Float64 = 0.1
comptime SGLD_NOISE_VAR: Float64 = 0.05

comptime NET = PCSequential[
    PCBlock[HIDDEN, HIDDEN, PCTanh],
    PCBlock[HIDDEN, DATA_DIM, PCTanh],
]
comptime TRAINER = PCTrainer[
    PCBlock[HIDDEN, HIDDEN, PCTanh],
    PCBlock[HIDDEN, DATA_DIM, PCTanh],
    dtype=dtype,
]
comptime OPT = PCAdam[LR=ADAM_LR]

# Bump per SGLD-step to keep substreams disjoint.
comptime PHILOX_BUMP_PER_STEP = UInt64(
    BATCH * NET.LATENT_DIM * (T_MIXING + T_SAMPLING) * 2
)


def _gauss_n01(mut rng: PhiloxRandom) -> Float64:
    """Box-Muller, returns one N(0,1) sample."""
    var u1 = Float64(rng.step_uniform()[0])
    var u2 = Float64(rng.step_uniform()[0])
    if u1 < 1e-10:
        u1 = 1e-10
    return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)


def main() raises:
    print("=" * 60)
    print("Stochastic tPC — roadmap Step 1")
    print("=" * 60)
    print("  arch       : PCBlock[", HIDDEN, ",", HIDDEN, ",PCTanh] → PCBlock[", HIDDEN, ",", DATA_DIM, ",PCTanh]")
    print("  PARAM_SIZE :", NET.PARAM_SIZE, "  LATENT_DIM:", NET.LATENT_DIM)
    print("  hyperparams: SEQ_LEN=", SEQ_LEN, " EPOCHS=", EPOCHS, " T_MIXING=", T_MIXING, " T_SAMPLING=", T_SAMPLING)
    print("  Phase A    : σ_obs=", OBS_NOISE_STD, "  SGLD noise_var=", SGLD_NOISE_VAR)
    print("  Phase B    : σ_obs=0  noise_var=0  (plain tPC parity)")

    var ds = MNIST()
    print("  [mnist] loaded:", MNIST.N_TRAIN, "train")

    # ── Build a fixed clean sequence (one image per digit class 0..SEQ_LEN-1) ──
    var seq_buf = alloc[Scalar[dtype]](SEQ_LEN * DATA_DIM).as_unsafe_any_origin()
    unsafe_memset(seq_buf, 0, SEQ_LEN * DATA_DIM)
    for digit in range(SEQ_LEN):
        var found_idx = -1
        for i in range(MNIST.N_TRAIN):
            if Int(ds.train_labels[i]) == digit:
                found_idx = i
                break
        if found_idx < 0:
            raise Error("digit " + String(digit) + " not found in MNIST")
        for j in range(DATA_DIM):
            seq_buf[digit * DATA_DIM + j] = ds.train_images[
                found_idx * DATA_DIM + j
            ]
    print("  [seq] built", SEQ_LEN, "clean images")

    # ── Allocate net params + Adam state (reused across phases via reinit) ────
    var params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var grads_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var opt_state_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE * OPT.STATE_PER_PARAM).as_unsafe_any_origin()
    var opt_global_buf = alloc[Scalar[dtype]](OPT.GLOBAL_STATE_SIZE).as_unsafe_any_origin()

    var params = tile[NET.PARAM_SIZE](params_buf)
    var grads = tile[NET.PARAM_SIZE](grads_buf)
    var opt_state = tile[NET.PARAM_SIZE, OPT.STATE_PER_PARAM](opt_state_buf)
    var opt_global = tile[OPT.GLOBAL_STATE_SIZE](opt_global_buf)

    # ── Scratch buffers (reused) ──────────────────────────────────────────────
    var lat_buf = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var mu_eps_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_OUT_DIM).as_unsafe_any_origin()
    var a_below_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var z_below_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var dx_buf_raw = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var noise_buf_raw = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()

    var latents = tile[BATCH, NET.LATENT_DIM](lat_buf)
    var mu_eps_buf = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_eps_buf_raw)
    var a_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](a_below_buf_raw)
    var z_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](z_below_buf_raw)
    var dx_buf = tile[BATCH, NET.LATENT_DIM](dx_buf_raw)
    var noise_buf = tile[BATCH, NET.LATENT_DIM](noise_buf_raw)

    # ── Per-step input + (possibly noisy) target buffers ──────────────────────
    var x_in_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var y_tgt_buf = alloc[Scalar[dtype]](BATCH * DATA_DIM).as_unsafe_any_origin()
    var x_in = tile[BATCH, HIDDEN](x_in_buf)
    var y_target = tile[BATCH, DATA_DIM](y_tgt_buf)

    # ── Recall-only buffers (built once, reused) ──────────────────────────────
    var recalls_buf = alloc[Scalar[dtype]](SEQ_LEN * DATA_DIM).as_unsafe_any_origin()
    var z_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var a_z_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var x_pred_buf = alloc[Scalar[dtype]](BATCH * DATA_DIM).as_unsafe_any_origin()
    var a_x_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var z_t_view = tile[BATCH, HIDDEN](z_buf)
    var a_z_view = tile[BATCH, HIDDEN](a_z_buf)
    var x_pred_view = tile[BATCH, DATA_DIM](x_pred_buf)
    var a_x_view = tile[BATCH, HIDDEN](a_x_buf)

    # ── Fixed hidden_init ~ N(0, 0.5²), shared across phases ──────────────────
    var hidden_init_buf = alloc[Scalar[dtype]](HIDDEN).as_unsafe_any_origin()
    var rng_init = PhiloxRandom(seed=UInt64(11), offset=UInt64(0))
    for i in range(HIDDEN):
        hidden_init_buf[i] = Scalar[dtype](0.5 * _gauss_n01(rng_init))

    comptime offset_b1 = NET._param_offset[1]()
    var params_b0 = tile[NET.block_types[0].PARAM_SIZE](params_buf)
    var params_b1 = tile[NET.block_types[1].PARAM_SIZE](params_buf + offset_b1)

    # ── Phase runner ─────────────────────────────────────────────────────────
    var ratio_a: Float64 = 0.0
    var ratio_b: Float64 = 0.0

    for phase in range(2):
        var is_stochastic = (phase == 0)
        var sigma_obs = OBS_NOISE_STD if is_stochastic else 0.0
        var sgld_var = SGLD_NOISE_VAR if is_stochastic else 0.0
        var phase_label = "A (stochastic)" if is_stochastic else "B (degenerate, noise_var=0)"

        print("\n" + "=" * 60)
        print("  Phase", phase_label)
        print("=" * 60)

        # Reset params + Adam + scratch.
        unsafe_memset(params_buf, 0, NET.PARAM_SIZE)
        unsafe_memset(grads_buf, 0, NET.PARAM_SIZE)
        unsafe_memset(opt_state_buf, 0, NET.PARAM_SIZE * OPT.STATE_PER_PARAM)
        unsafe_memset(opt_global_buf, 0, OPT.GLOBAL_STATE_SIZE)
        unsafe_memset(lat_buf, 0, BATCH * NET.LATENT_DIM)
        unsafe_memset(mu_eps_buf_raw, 0, BATCH * NET.SCRATCH_OUT_DIM)
        unsafe_memset(a_below_buf_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
        unsafe_memset(z_below_buf_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
        unsafe_memset(dx_buf_raw, 0, BATCH * NET.LATENT_DIM)
        unsafe_memset(noise_buf_raw, 0, BATCH * NET.LATENT_DIM)
        NET.pc_init_params[PCXavier, dtype](params)

        # Per-phase RNGs: one for obs corruption, one for SGLD substream offset.
        var obs_rng = PhiloxRandom(seed=UInt64(23 + phase), offset=UInt64(0))
        var noise_offset = UInt64(1_000_000) + UInt64(phase) * UInt64(500_000_000)
        var philox_seed = UInt64(42 + phase)

        print("\n  epoch | last_step_loss | wall_t (s)")
        print("  ------+----------------+------------")
        var step_num: Int = 0
        var t0 = perf_counter_ns()

        for epoch in range(EPOCHS):
            # Reset prev_hidden = hidden_init at the start of each epoch.
            for j in range(HIDDEN):
                x_in_buf[j] = hidden_init_buf[j]

            var last_loss: Float64 = 0.0
            for t in range(SEQ_LEN):
                # y_target = data[t] + ε_obs (ε_obs = 0 in Phase B).
                for j in range(DATA_DIM):
                    var noise = sigma_obs * _gauss_n01(obs_rng) if is_stochastic else 0.0
                    y_tgt_buf[j] = seq_buf[t * DATA_DIM + j] + Scalar[dtype](noise)

                var result = TRAINER.compute_grads_only_mcpc[BATCH](
                    params, grads, latents,
                    mu_eps_buf, a_below_buf, z_below_buf, dx_buf, noise_buf,
                    x_in, y_target,
                    T_mixing=T_MIXING,
                    T_sampling=T_SAMPLING,
                    lr_x=Scalar[dtype](LR_X),
                    noise_var=Scalar[dtype](sgld_var),
                    seed=philox_seed,
                    offset_base=noise_offset,
                )
                noise_offset += PHILOX_BUMP_PER_STEP

                step_num += 1
                OPT.step[NET.PARAM_SIZE, dtype](
                    params, grads, opt_state, opt_global, step_num
                )
                last_loss = result.output_loss_final

                # Save z_t (only interior latent here) → x_in for next step.
                for j in range(HIDDEN):
                    x_in_buf[j] = lat_buf[j]

            if epoch == 0 or (epoch + 1) % 20 == 0 or epoch == EPOCHS - 1:
                var elapsed = Float64(perf_counter_ns() - t0) / 1e9
                print(
                    "    ", epoch, "  ",
                    fit(String(last_loss), 11), "  ",
                    fit(String(elapsed), 7),
                )

        var total_t = Float64(perf_counter_ns() - t0) / 1e9
        print("\n  total train time:", total_t, "s")

        # ── Recall — settle z_0 against CLEAN data[0] (no obs noise at recall) ─
        for j in range(HIDDEN):
            x_in_buf[j] = hidden_init_buf[j]
        for j in range(DATA_DIM):
            y_tgt_buf[j] = seq_buf[0 * DATA_DIM + j]
        _ = TRAINER.compute_grads_only_mcpc[BATCH](
            params, grads, latents,
            mu_eps_buf, a_below_buf, z_below_buf, dx_buf, noise_buf,
            x_in, y_target,
            T_mixing=T_MIXING,
            T_sampling=T_SAMPLING,
            lr_x=Scalar[dtype](LR_X),
            noise_var=Scalar[dtype](0.0),  # deterministic settle at recall
            seed=philox_seed,
            offset_base=noise_offset,
        )
        # No Adam.step — params frozen for recall.
        for j in range(HIDDEN):
            x_in_buf[j] = lat_buf[j]

        unsafe_memset(recalls_buf, 0, SEQ_LEN * DATA_DIM)
        for j in range(DATA_DIM):
            recalls_buf[0 * DATA_DIM + j] = seq_buf[0 * DATA_DIM + j]

        for t in range(1, SEQ_LEN):
            NET.block_types[0].predict[BATCH, dtype](
                x_in, params_b0, z_t_view, a_z_view
            )
            NET.block_types[1].predict[BATCH, dtype](
                z_t_view, params_b1, x_pred_view, a_x_view
            )
            for j in range(DATA_DIM):
                recalls_buf[t * DATA_DIM + j] = x_pred_buf[j]
            for j in range(HIDDEN):
                x_in_buf[j] = z_buf[j]

        # ── MSE per recalled image vs CLEAN data + zero baseline ──────────────
        print("\n  step | mse(recall_t, clean_t) | mse(zeros, clean_t)")
        print("  -----+------------------------+-------------------")
        var total_recall_mse: Float64 = 0.0
        var total_zero_mse: Float64 = 0.0
        for t in range(1, SEQ_LEN):
            var r_mse: Float64 = 0.0
            var z_mse: Float64 = 0.0
            for j in range(DATA_DIM):
                var d = Float64(recalls_buf[t * DATA_DIM + j]) - Float64(
                    seq_buf[t * DATA_DIM + j]
                )
                r_mse += d * d
                var d2 = Float64(seq_buf[t * DATA_DIM + j])
                z_mse += d2 * d2
            r_mse /= Float64(DATA_DIM)
            z_mse /= Float64(DATA_DIM)
            total_recall_mse += r_mse
            total_zero_mse += z_mse
            print("    ", t, "  ", r_mse, "    ", z_mse)

        var avg_recall = total_recall_mse / Float64(SEQ_LEN - 1)
        var avg_zero = total_zero_mse / Float64(SEQ_LEN - 1)
        var ratio = avg_recall / avg_zero if avg_zero > 0 else 1.0
        print("\n  avg recall MSE :", avg_recall)
        print("  avg zero   MSE :", avg_zero)
        print("  recall / zero  :", ratio)
        if phase == 0:
            ratio_a = ratio
        else:
            ratio_b = ratio

    # ── Pass criteria ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print("  Phase A ratio (stochastic):", ratio_a)
    print("  Phase B ratio (degenerate):", ratio_b)

    var pass_a = ratio_a < 0.7
    var pass_b = ratio_b < 0.7

    if pass_a and pass_b:
        print("\n  [PASS] Stochastic tPC: both phases beat zero baseline by ≥30%")
    else:
        if not pass_a:
            print("\n  [FAIL] Phase A (stochastic) ratio", ratio_a, "≥ 0.7")
        if not pass_b:
            print("\n  [FAIL] Phase B (degenerate) ratio", ratio_b, "≥ 0.7")
        raise Error("stochastic tPC test failed")

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
    y_tgt_buf.free()
    seq_buf.free()
    hidden_init_buf.free()
    recalls_buf.free()
    z_buf.free()
    a_z_buf.free()
    x_pred_buf.free()
    a_x_buf.free()
    print("=== Done ===")

from noeira.core.fmt import fit