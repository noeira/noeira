"""tPC sequence-memory test — Bogacz notebook 4 reproduction (CPU).

Trains a recurrent PC network to memorize a sequence of MNIST digits, then
recalls the sequence given just the initial hidden state + the first image.

Architecture (mirrors notebook 4 with smaller hidden):
    PCBlock[hidden, hidden, PCTanh]   # block_0 — recurrent: μ_z = W_r·tanh(z_{t-1}) + b_0
    PCBlock[hidden, 784,   PCTanh]    # block_1 — decoder:  μ_x = W_dec·tanh(z_t) + b_dec

Note: Our framework's natural recurrence puts Tanh on the input side of each
block, giving z_t = W_r·tanh(z_{t-1}) + b. Bogacz's notebook 4 has the Tanh
*outside* W_r (i.e. z_t = W_r·z_{t-1} with a separate Tanh after the latent).
Both are valid recurrent dynamics; ours is simpler with the existing PCBlock.

Training loop per epoch:
    prev_hidden = hidden_init  (random, fixed across epochs)
    for t in 0..seq_len-1:
        x_in = prev_hidden
        y_target = data[t]
        compute_grads_only(...) + Adam.step(...)
        prev_hidden = latents[:, 0:hidden]   # z_t after settling

Recall:
    Settle z_0 against data[0] via inference (no weight update).
    For t = 1..seq_len-1:
        z_t   = block_0.predict(prev_hidden)
        x_t   = block_1.predict(z_t)
        prev_hidden = z_t

Pass criterion: average per-image MSE over recalled images is at least 30%
better than a zero-prediction baseline (= mean pixel² across data).

Run:
    pixi run mojo run -I . tests/pcn/test_tpc_mnist_seq.mojo
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


comptime BATCH = 1                 # one sequence at a time (Bogacz default)
comptime HIDDEN = 64
comptime DATA_DIM = 784
comptime SEQ_LEN = 5               # smaller than Bogacz's 10 for CPU speed
comptime EPOCHS = 100
comptime T_INFER = 50
comptime LR_X: Float64 = 0.01
comptime ADAM_LR: Float64 = 0.005

comptime NET = PCSequential[
    PCBlock[HIDDEN, HIDDEN, PCTanh],     # recurrent block
    PCBlock[HIDDEN, DATA_DIM, PCTanh],   # decoder block
]
comptime TRAINER = PCTrainer[
    PCBlock[HIDDEN, HIDDEN, PCTanh],
    PCBlock[HIDDEN, DATA_DIM, PCTanh],
    dtype=dtype,
]
comptime OPT = PCAdam[LR=ADAM_LR]


def main() raises:
    print("=" * 60)
    print("tPC sequence memory — Bogacz notebook 4 reproduction (CPU)")
    print("=" * 60)
    print("  arch       : PCBlock[", HIDDEN, ",", HIDDEN, ",PCTanh] → PCBlock[", HIDDEN, ",", DATA_DIM, ",PCTanh]")
    print("  PARAM_SIZE :", NET.PARAM_SIZE)
    print("  LATENT_DIM :", NET.LATENT_DIM)
    print("  SEQ_LEN    :", SEQ_LEN, " HIDDEN=", HIDDEN)
    print("  hyperparams: T_INFER=", T_INFER, " EPOCHS=", EPOCHS)
    print("  Adam lr    :", ADAM_LR, "  lr_x=", LR_X)

    var ds = MNIST()
    print("  [mnist] loaded:", MNIST.N_TRAIN, "train")

    # ── Build sequence: one image of each of the first SEQ_LEN digits ────────
    var seq_buf = alloc[Scalar[dtype]](SEQ_LEN * DATA_DIM).as_unsafe_any_origin()
    unsafe_memset(seq_buf, 0, SEQ_LEN * DATA_DIM)
    for digit in range(SEQ_LEN):
        # Find the first image of this digit class in train set
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
    print("  [seq] built", SEQ_LEN, "images, one per digit")

    # ── Allocate net params + Adam state ─────────────────────────────────────
    var params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var grads_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var opt_state_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE * OPT.STATE_PER_PARAM).as_unsafe_any_origin()
    var opt_global_buf = alloc[Scalar[dtype]](OPT.GLOBAL_STATE_SIZE).as_unsafe_any_origin()
    unsafe_memset(params_buf, 0, NET.PARAM_SIZE)
    unsafe_memset(grads_buf, 0, NET.PARAM_SIZE)
    unsafe_memset(opt_state_buf, 0, NET.PARAM_SIZE * OPT.STATE_PER_PARAM)
    unsafe_memset(opt_global_buf, 0, OPT.GLOBAL_STATE_SIZE)

    var params = tile[NET.PARAM_SIZE](params_buf)
    var grads = tile[NET.PARAM_SIZE](grads_buf)
    var opt_state = tile[NET.PARAM_SIZE, OPT.STATE_PER_PARAM](opt_state_buf)
    var opt_global = tile[OPT.GLOBAL_STATE_SIZE](opt_global_buf)
    NET.pc_init_params[PCXavier, dtype](params)

    # ── Scratch buffers ──────────────────────────────────────────────────────
    var lat_buf = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var mu_eps_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_OUT_DIM).as_unsafe_any_origin()
    var a_below_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var z_below_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var dx_buf_raw = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    unsafe_memset(lat_buf, 0, BATCH * NET.LATENT_DIM)
    unsafe_memset(mu_eps_buf_raw, 0, BATCH * NET.SCRATCH_OUT_DIM)
    unsafe_memset(a_below_buf_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
    unsafe_memset(z_below_buf_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
    unsafe_memset(dx_buf_raw, 0, BATCH * NET.LATENT_DIM)

    var latents = tile[BATCH, NET.LATENT_DIM](lat_buf)
    var mu_eps_buf = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_eps_buf_raw)
    var a_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](a_below_buf_raw)
    var z_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](z_below_buf_raw)
    var dx_buf = tile[BATCH, NET.LATENT_DIM](dx_buf_raw)

    # ── Per-step input + target buffers ──────────────────────────────────────
    var x_in_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var y_tgt_buf = alloc[Scalar[dtype]](BATCH * DATA_DIM).as_unsafe_any_origin()
    unsafe_memset(x_in_buf, 0, BATCH * HIDDEN)
    unsafe_memset(y_tgt_buf, 0, BATCH * DATA_DIM)
    var x_in = tile[BATCH, HIDDEN](x_in_buf)
    var y_target = tile[BATCH, DATA_DIM](y_tgt_buf)

    # ── Generate fixed hidden_init ~ N(0, 0.5²) ──────────────────────────────
    # (Gaussian init like Bogacz's torch.randn, scaled smaller to keep tanh
    # away from saturation initially.)
    var hidden_init_buf = alloc[Scalar[dtype]](HIDDEN).as_unsafe_any_origin()
    var rng = PhiloxRandom(seed=UInt64(11), offset=UInt64(0))
    for i in range(HIDDEN):
        var u1 = rng.step_uniform()[0]
        var u2 = rng.step_uniform()[0]
        if u1 < 1e-10:
            u1 = 1e-10
        var r = sqrt(-2.0 * log(Float64(u1)))
        var z = r * cos(2.0 * pi * Float64(u2))
        hidden_init_buf[i] = Scalar[dtype](0.5 * z)

    # ── Training loop: per epoch, walk through the sequence ──────────────────
    print("\n  epoch | last_step_loss | wall_t (s)")
    print("  ------+----------------+------------")
    var step_num: Int = 0
    var t0 = perf_counter_ns()

    for epoch in range(EPOCHS):
        # Reset prev_hidden to hidden_init at the start of each epoch
        for j in range(HIDDEN):
            x_in_buf[j] = hidden_init_buf[j]

        var last_loss: Float64 = 0.0
        for t in range(SEQ_LEN):
            # Set y_target = data[t]
            for j in range(DATA_DIM):
                y_tgt_buf[j] = seq_buf[t * DATA_DIM + j]

            var result = TRAINER.compute_grads_only[BATCH](
                params, grads, latents,
                mu_eps_buf, a_below_buf, z_below_buf, dx_buf,
                x_in, y_target,
                T_infer=T_INFER,
                lr_x=Scalar[dtype](LR_X),
            )
            step_num += 1
            OPT.step[NET.PARAM_SIZE, dtype](
                params, grads, opt_state, opt_global, step_num
            )
            last_loss = result.output_loss_final
            # Save z_t (the only interior latent) → x_in for next step
            for j in range(HIDDEN):
                x_in_buf[j] = lat_buf[j]

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            var elapsed = Float64(perf_counter_ns() - t0) / 1e9
            print(
                "    ", epoch, "  ",
                fit(String(last_loss), 11), "  ",
                fit(String(elapsed), 7),
            )

    var total_t = Float64(perf_counter_ns() - t0) / 1e9
    print("\n  total train time:", total_t, "s")

    # ── Recall ──────────────────────────────────────────────────────────────
    # Step 0: settle z_0 against data[0] via inference (no weight update).
    for j in range(HIDDEN):
        x_in_buf[j] = hidden_init_buf[j]
    for j in range(DATA_DIM):
        y_tgt_buf[j] = seq_buf[0 * DATA_DIM + j]
    _ = TRAINER.compute_grads_only[BATCH](
        params, grads, latents,
        mu_eps_buf, a_below_buf, z_below_buf, dx_buf,
        x_in, y_target,
        T_infer=T_INFER,
        lr_x=Scalar[dtype](LR_X),
    )
    # No Adam.step for recall (params frozen).
    # prev_hidden = z_0 (saved into x_in_buf for next step's predict)
    for j in range(HIDDEN):
        x_in_buf[j] = lat_buf[j]

    # Steps 1..SEQ_LEN-1: feedforward only — predict next image from prev_hidden.
    var recalls_buf = alloc[Scalar[dtype]](SEQ_LEN * DATA_DIM).as_unsafe_any_origin()
    unsafe_memset(recalls_buf, 0, SEQ_LEN * DATA_DIM)
    # First image is just the input (recall starts at t=1).
    for j in range(DATA_DIM):
        recalls_buf[0 * DATA_DIM + j] = seq_buf[0 * DATA_DIM + j]

    var z_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var a_z_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var x_pred_buf = alloc[Scalar[dtype]](BATCH * DATA_DIM).as_unsafe_any_origin()
    var a_x_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var z_t = tile[BATCH, HIDDEN](z_buf)
    var a_z_t = tile[BATCH, HIDDEN](a_z_buf)
    var x_pred_t = tile[BATCH, DATA_DIM](x_pred_buf)
    var a_x_t = tile[BATCH, HIDDEN](a_x_buf)

    # Get param views per block
    comptime offset_b1 = NET._param_offset[1]()
    var params_b0 = tile[NET.block_types[0].PARAM_SIZE](params_buf)
    var params_b1 = tile[NET.block_types[1].PARAM_SIZE](params_buf + offset_b1)

    for t in range(1, SEQ_LEN):
        # block_0.predict(prev_hidden) → z_t
        NET.block_types[0].predict[BATCH, dtype](
            x_in, params_b0, z_t, a_z_t
        )
        # block_1.predict(z_t) → x_pred (the recalled image)
        NET.block_types[1].predict[BATCH, dtype](
            z_t, params_b1, x_pred_t, a_x_t
        )
        for j in range(DATA_DIM):
            recalls_buf[t * DATA_DIM + j] = x_pred_buf[j]
        # prev_hidden = z_t
        for j in range(HIDDEN):
            x_in_buf[j] = z_buf[j]

    # ── MSE per recalled image + zero-prediction baseline ────────────────────
    print("\n  step | mse(recall_t, data_t) | mse(zeros, data_t)")
    print("  -----+-----------------------+-----------------")
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

    if ratio < 0.7:
        print("\n  [PASS] tPC recalls the sequence (≥30% better than zero baseline)")
    else:
        print("\n  [FAIL] recall not significantly better than zero baseline")
        raise Error("tPC test failed: recall ratio " + String(ratio))

    params_buf.free()
    grads_buf.free()
    opt_state_buf.free()
    opt_global_buf.free()
    lat_buf.free()
    mu_eps_buf_raw.free()
    a_below_buf_raw.free()
    z_below_buf_raw.free()
    dx_buf_raw.free()
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