"""Stochastic tPC (GPU) — Step 1 of PCN_WORLD_MODEL_ROADMAP.md.

GPU port of `test_stochastic_tpc.mojo`. Same architecture, two phases (A:
stochastic, B: degenerate), same pass criterion (recall ratio < 0.7 on both).

Uses `compute_grads_only_mcpc_gpu` from pc_trainer.mojo. Per training step,
y_target = clean_data[t] + obs_noise is built on host (BATCH=1, DATA_DIM=784
is tiny) and uploaded.

Run:
    pixi run -e apple  mojo run -I . tests/pcn/test_stochastic_tpc_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/pcn/test_stochastic_tpc_gpu.mojo
"""

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from std.math import sqrt, log, cos, pi
from std.random.philox import Random as PhiloxRandom
from std.time import perf_counter_ns
from noeira.core.tiles import Tile, tile

from noeira.nn.constants import DT as dtype
from noeira.experimental.pcn.pc_constants import TPB
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

comptime PHILOX_BUMP_PER_STEP = UInt64(
    BATCH * NET.LATENT_DIM * (T_MIXING + T_SAMPLING) * 2
)


def _gauss_n01(mut rng: PhiloxRandom) -> Float64:
    var u1 = Float64(rng.step_uniform()[0])
    var u2 = Float64(rng.step_uniform()[0])
    if u1 < 1e-10:
        u1 = 1e-10
    return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)


def _copy_lat_to_x_in_kernel[
    BATCH: Int,
    HIDDEN: Int,
    LATENT_DIM: Int,
    KDT: DType,
](
    latents: Tile[KDT, BATCH, LATENT_DIM],
    x_in: Tile[KDT, BATCH, HIDDEN],
):
    """Copy lat[:, 0:HIDDEN] → x_in[:, 0:HIDDEN]. Here IN_DIM == HIDDEN."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * HIDDEN:
        return
    var b = idx // HIDDEN
    var k = idx % HIDDEN
    x_in[b, k] = rebind[Scalar[KDT]](latents[b, k])


def main() raises:
    print("=" * 60)
    print("Stochastic tPC (GPU) — roadmap Step 1")
    print("=" * 60)
    print(
        "  arch       : PCBlock[",
        HIDDEN,
        ",",
        HIDDEN,
        ",PCTanh] → PCBlock[",
        HIDDEN,
        ",",
        DATA_DIM,
        ",PCTanh]",
    )
    print("  PARAM_SIZE :", NET.PARAM_SIZE, "  LATENT_DIM:", NET.LATENT_DIM)
    print(
        "  Phase A    : σ_obs=",
        OBS_NOISE_STD,
        "  SGLD noise_var=",
        SGLD_NOISE_VAR,
    )
    print("  Phase B    : σ_obs=0  noise_var=0  (plain tPC parity)")

    var ctx = DeviceContext()
    var ds = MNIST()
    print("  [mnist] loaded:", MNIST.N_TRAIN, "train")

    # ── Build clean sequence on host (one image per digit class) ──────────────
    var seq_host = ctx.enqueue_create_host_buffer[dtype](SEQ_LEN * DATA_DIM)
    for i in range(SEQ_LEN * DATA_DIM):
        seq_host.unsafe_ptr()[i] = Scalar[dtype](0)
    for digit in range(SEQ_LEN):
        var found_idx = -1
        for i in range(MNIST.N_TRAIN):
            if Int(ds.train_labels[i]) == digit:
                found_idx = i
                break
        if found_idx < 0:
            raise Error("digit " + String(digit) + " not found in MNIST")
        for j in range(DATA_DIM):
            seq_host.unsafe_ptr()[digit * DATA_DIM + j] = ds.train_images[
                found_idx * DATA_DIM + j
            ]
    print("  [seq] built", SEQ_LEN, "clean images")

    # ── Allocate GPU buffers (reused across phases) ───────────────────────────
    var params_dbuf = ctx.enqueue_create_buffer[dtype](NET.PARAM_SIZE)
    var grads_dbuf = ctx.enqueue_create_buffer[dtype](NET.PARAM_SIZE)
    var lat_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.LATENT_DIM)
    var mu_eps_dbuf = ctx.enqueue_create_buffer[dtype](
        BATCH * NET.SCRATCH_OUT_DIM
    )
    var a_below_dbuf = ctx.enqueue_create_buffer[dtype](
        BATCH * NET.SCRATCH_IN_DIM
    )
    var z_below_dbuf = ctx.enqueue_create_buffer[dtype](
        BATCH * NET.SCRATCH_IN_DIM
    )
    var dx_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.LATENT_DIM)
    var noise_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.LATENT_DIM)
    var x_in_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    var y_tgt_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * DATA_DIM)
    var opt_state_dbuf = ctx.enqueue_create_buffer[dtype](
        NET.PARAM_SIZE * OPT.STATE_PER_PARAM
    )
    var opt_global_dbuf = ctx.enqueue_create_buffer[dtype](
        OPT.GLOBAL_STATE_SIZE
    )

    var params_t = tile[NET.PARAM_SIZE](params_dbuf)
    var grads_t = tile[NET.PARAM_SIZE](grads_dbuf)
    var lat_t = tile[BATCH, NET.LATENT_DIM](lat_dbuf)
    var mu_eps_t = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_eps_dbuf)
    var a_below_t = tile[BATCH, NET.SCRATCH_IN_DIM](a_below_dbuf)
    var z_below_t = tile[BATCH, NET.SCRATCH_IN_DIM](z_below_dbuf)
    var dx_t = tile[BATCH, NET.LATENT_DIM](dx_dbuf)
    var noise_t = tile[BATCH, NET.LATENT_DIM](noise_dbuf)
    var x_in_t = tile[BATCH, HIDDEN](x_in_dbuf)
    var y_tgt_t = tile[BATCH, DATA_DIM](y_tgt_dbuf)
    var opt_state_t = tile[NET.PARAM_SIZE, OPT.STATE_PER_PARAM](opt_state_dbuf)
    var opt_global_t = tile[OPT.GLOBAL_STATE_SIZE](opt_global_dbuf)

    # Host staging buffers
    var params_init_host = ctx.enqueue_create_host_buffer[dtype](NET.PARAM_SIZE)
    var x_in_host = ctx.enqueue_create_host_buffer[dtype](BATCH * HIDDEN)
    var y_tgt_host = ctx.enqueue_create_host_buffer[dtype](BATCH * DATA_DIM)
    var opt_state_init_host = ctx.enqueue_create_host_buffer[dtype](
        NET.PARAM_SIZE * OPT.STATE_PER_PARAM
    )
    var opt_global_init_host = ctx.enqueue_create_host_buffer[dtype](
        OPT.GLOBAL_STATE_SIZE
    )
    var lat_host = ctx.enqueue_create_host_buffer[dtype](BATCH * NET.LATENT_DIM)
    var pred_host = ctx.enqueue_create_host_buffer[dtype](BATCH * DATA_DIM)

    # ── Fixed hidden_init ~ N(0, 0.5²), shared across phases ──────────────────
    var hidden_init_host = ctx.enqueue_create_host_buffer[dtype](HIDDEN)
    var rng_init = PhiloxRandom(seed=UInt64(11), offset=UInt64(0))
    for i in range(HIDDEN):
        hidden_init_host.unsafe_ptr()[i] = Scalar[dtype](
            0.5 * _gauss_n01(rng_init)
        )

    # ── Per-block param views for eval feedforward ────────────────────────────
    comptime offset_b1 = NET._param_offset[1]()
    var params_b0_t = tile[NET.block_types[0].PARAM_SIZE](params_dbuf.unsafe_ptr().as_unsafe_any_origin())
    var params_b1_t = tile[NET.block_types[1].PARAM_SIZE](params_dbuf.unsafe_ptr().as_unsafe_any_origin() + offset_b1)

    # Eval feedforward scratch
    var z_pred_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    var a_z_pred_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    var x_pred_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * DATA_DIM)
    var a_x_pred_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    var z_pred_t = tile[BATCH, HIDDEN](z_pred_dbuf)
    var a_z_pred_t = tile[BATCH, HIDDEN](a_z_pred_dbuf)
    var x_pred_t = tile[BATCH, DATA_DIM](x_pred_dbuf)
    var a_x_pred_t = tile[BATCH, HIDDEN](a_x_pred_dbuf)

    var ratio_a: Float64 = 0.0
    var ratio_b: Float64 = 0.0

    for phase in range(2):
        var is_stochastic = phase == 0
        var sigma_obs = OBS_NOISE_STD if is_stochastic else 0.0
        var sgld_var = SGLD_NOISE_VAR if is_stochastic else 0.0
        var phase_label = (
            "A (stochastic)" if is_stochastic else "B (degenerate, noise_var=0)"
        )

        print("\n" + "=" * 60)
        print("  Phase", phase_label)
        print("=" * 60)

        # Reinit params + Adam state on host, upload.
        for i in range(NET.PARAM_SIZE):
            params_init_host.unsafe_ptr()[i] = Scalar[dtype](0)
        var params_init_t = tile[NET.PARAM_SIZE](params_init_host.unsafe_ptr().as_unsafe_any_origin())
        NET.pc_init_params[PCXavier, dtype](params_init_t)
        ctx.enqueue_copy(params_dbuf, params_init_host)

        for i in range(NET.PARAM_SIZE * OPT.STATE_PER_PARAM):
            opt_state_init_host.unsafe_ptr()[i] = Scalar[dtype](0)
        ctx.enqueue_copy(opt_state_dbuf, opt_state_init_host)

        opt_global_init_host.unsafe_ptr()[0] = Scalar[dtype](0)
        opt_global_init_host.unsafe_ptr()[1] = Scalar[dtype](1.0)
        ctx.enqueue_copy(opt_global_dbuf, opt_global_init_host)

        var obs_rng = PhiloxRandom(seed=UInt64(23 + phase), offset=UInt64(0))
        var noise_offset = UInt64(1_000_000) + UInt64(phase) * UInt64(
            500_000_000
        )
        var philox_seed = UInt64(42 + phase)

        var step_num: Int = 0
        var t0 = perf_counter_ns()

        print("\n  epoch | wall_t (s)")
        print("  ------+-----------")

        for epoch in range(EPOCHS):
            # Reset prev_hidden = hidden_init.
            for j in range(HIDDEN):
                x_in_host.unsafe_ptr()[j] = hidden_init_host.unsafe_ptr()[j]
            ctx.enqueue_copy(x_in_dbuf, x_in_host)

            for t in range(SEQ_LEN):
                # y_target = data[t] + ε_obs
                for j in range(DATA_DIM):
                    var noise = (
                        sigma_obs
                        * _gauss_n01(obs_rng) if is_stochastic else 0.0
                    )
                    y_tgt_host.unsafe_ptr()[j] = seq_host.unsafe_ptr()[
                        t * DATA_DIM + j
                    ] + Scalar[dtype](noise)
                ctx.enqueue_copy(y_tgt_dbuf, y_tgt_host)

                TRAINER.compute_grads_only_mcpc_gpu[BATCH](
                    ctx,
                    params_t,
                    grads_t,
                    lat_t,
                    mu_eps_t,
                    a_below_t,
                    z_below_t,
                    dx_t,
                    noise_t,
                    x_in_t,
                    y_tgt_t,
                    T_mixing=T_MIXING,
                    T_sampling=T_SAMPLING,
                    lr_x=Scalar[dtype](LR_X),
                    noise_var=Scalar[dtype](sgld_var),
                    seed=philox_seed,
                    offset_base=noise_offset,
                )
                noise_offset += PHILOX_BUMP_PER_STEP

                step_num += 1
                OPT.step_gpu[NET.PARAM_SIZE, dtype](
                    ctx, params_t, grads_t, opt_state_t, opt_global_t, step_num
                )

                # prev_hidden = lat[:, 0:HIDDEN]  (HIDDEN == LATENT_DIM here)
                comptime cp_k = _copy_lat_to_x_in_kernel[
                    BATCH, HIDDEN, NET.LATENT_DIM, dtype
                ]
                var cp_threads = BATCH * HIDDEN
                var cp_blocks = (cp_threads + TPB - 1) // TPB
                ctx.enqueue_function[cp_k](
                    lat_t,
                    x_in_t,
                    grid_dim=(cp_blocks,),
                    block_dim=(TPB,),
                )

            if epoch == 0 or (epoch + 1) % 25 == 0 or epoch == EPOCHS - 1:
                ctx.synchronize()
                var elapsed = Float64(perf_counter_ns() - t0) / 1e9
                print("    ", epoch, "  ", fit(String(elapsed), 7))

        ctx.synchronize()
        var total_t = Float64(perf_counter_ns() - t0) / 1e9
        print("\n  total train time:", total_t, "s")

        # ── Recall: settle z_0 against CLEAN data[0] (deterministic) ──────────
        for j in range(HIDDEN):
            x_in_host.unsafe_ptr()[j] = hidden_init_host.unsafe_ptr()[j]
        ctx.enqueue_copy(x_in_dbuf, x_in_host)
        for j in range(DATA_DIM):
            y_tgt_host.unsafe_ptr()[j] = seq_host.unsafe_ptr()[0 * DATA_DIM + j]
        ctx.enqueue_copy(y_tgt_dbuf, y_tgt_host)

        TRAINER.compute_grads_only_mcpc_gpu[BATCH](
            ctx,
            params_t,
            grads_t,
            lat_t,
            mu_eps_t,
            a_below_t,
            z_below_t,
            dx_t,
            noise_t,
            x_in_t,
            y_tgt_t,
            T_mixing=T_MIXING,
            T_sampling=T_SAMPLING,
            lr_x=Scalar[dtype](LR_X),
            noise_var=Scalar[dtype](0.0),  # deterministic settle
            seed=philox_seed,
            offset_base=noise_offset,
        )
        # No Adam.step.

        # prev_hidden = settled z_0
        comptime cp_k_recall = _copy_lat_to_x_in_kernel[
            BATCH, HIDDEN, NET.LATENT_DIM, dtype
        ]
        var cp_threads_recall = BATCH * HIDDEN
        var cp_blocks_recall = (cp_threads_recall + TPB - 1) // TPB
        ctx.enqueue_function[cp_k_recall](
            lat_t,
            x_in_t,
            grid_dim=(cp_blocks_recall,),
            block_dim=(TPB,),
        )

        # Recall steps 1..SEQ_LEN-1: feedforward only.
        var total_recall_mse: Float64 = 0.0
        var total_zero_mse: Float64 = 0.0
        print("\n  step | mse(recall_t, clean_t) | mse(zeros, clean_t)")
        print("  -----+------------------------+-------------------")

        # Step 0 = data itself (baseline).
        # For t >= 1: predict via feedforward.
        for t in range(1, SEQ_LEN):
            NET.block_types[0].predict_gpu[BATCH, dtype](
                ctx, x_in_t, params_b0_t, z_pred_t, a_z_pred_t
            )
            NET.block_types[1].predict_gpu[BATCH, dtype](
                ctx, z_pred_t, params_b1_t, x_pred_t, a_x_pred_t
            )

            ctx.enqueue_copy(pred_host, x_pred_dbuf)
            ctx.synchronize()

            var r_mse: Float64 = 0.0
            var z_mse: Float64 = 0.0
            for j in range(DATA_DIM):
                var d = Float64(pred_host.unsafe_ptr()[j]) - Float64(
                    seq_host.unsafe_ptr()[t * DATA_DIM + j]
                )
                r_mse += d * d
                var d2 = Float64(seq_host.unsafe_ptr()[t * DATA_DIM + j])
                z_mse += d2 * d2
            r_mse /= Float64(DATA_DIM)
            z_mse /= Float64(DATA_DIM)
            total_recall_mse += r_mse
            total_zero_mse += z_mse
            print("    ", t, "  ", r_mse, "    ", z_mse)

            # prev_hidden = z_pred (the predicted next latent)
            ctx.enqueue_copy(lat_host, z_pred_dbuf)
            ctx.synchronize()
            for j in range(HIDDEN):
                x_in_host.unsafe_ptr()[j] = lat_host.unsafe_ptr()[j]
            ctx.enqueue_copy(x_in_dbuf, x_in_host)

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
        print(
            "\n  [PASS] Stochastic tPC GPU: both phases beat zero baseline by"
            " ≥30%"
        )
    else:
        if not pass_a:
            print("\n  [FAIL] Phase A (stochastic) ratio", ratio_a, "≥ 0.7")
        if not pass_b:
            print("\n  [FAIL] Phase B (degenerate) ratio", ratio_b, "≥ 0.7")
        raise Error("stochastic tPC GPU test failed")

    print("=== Done ===")

from noeira.core.fmt import fit