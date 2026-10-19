"""Bidirectional PC test — Bogacz notebook 5 reproduction (CPU).

Trains two PC paths sharing two common latents (x0, x1):
  - UP   : image (784) → x0 (256) → x1 (256) → label (10)
  - DOWN : label (10)  → x1 (256) → x0 (256) → image (784)

Total energy = alpha_up * E_up + alpha_down * E_down.
Each shared latent is updated by the SUM of both paths' gradients per
inference iteration.

Architecture matches Bogacz notebook 5:
    UP  : PCBlock[784,256,Id] → PCBlock[256,256,ReLU] → PCBlock[256,10,ReLU]
    DOWN: PCBlock[10,256,Id]  → PCBlock[256,256,ReLU] → PCBlock[256,784,ReLU]

Pass criteria:
  - Up classification accuracy ≥ 70% on test subset.
  - Generated per-class images differ across classes.

Run:
    pixi run mojo run -I . tests/pcn/test_bidirectional_pc.mojo
"""

from std.memory import alloc, memset
from std.time import perf_counter_ns
from std.math import exp
from noeira.core.tiles import tile

from noeira.nn.constants import DT as dtype
from noeira.experimental.pcn.pc_initializer import PCXavier
from noeira.experimental.pcn.pc_optimizer import PCAdam
from noeira.nn.datasets.mnist import MNIST
from noeira.experimental.pcn import (
    PCBlock,
    PCSequential,
    PCIdentity,
    PCReLU,
)
from noeira.render.image_writer import save_image_row, save_reconstruction_grid


# ── Notebook 5 hyperparameters ──────────────────────────────────────────────
comptime BATCH = 500
comptime HIDDEN = 256
comptime EPOCHS = 10
comptime T_INFER = 20
comptime LR_X: Float64 = 0.01
comptime ADAM_LR: Float64 = 0.01
comptime ALPHA_UP: Float64 = 1.0
comptime ALPHA_DOWN: Float64 = 0.0001

comptime N_TRAIN = 10000
comptime N_TEST = 1000
comptime N_TRAIN_BATCHES = N_TRAIN // BATCH
comptime N_TEST_BATCHES = N_TEST // BATCH

# ── UP path: image → x0 → x1 → label ──────────────────────────────────────
comptime UB0 = PCBlock[784, HIDDEN, PCIdentity]
comptime UB1 = PCBlock[HIDDEN, HIDDEN, PCReLU]
comptime UB2 = PCBlock[HIDDEN, 10, PCReLU]
comptime UP_NET = PCSequential[UB0, UB1, UB2]
comptime UP_PARAM_SIZE = UP_NET.PARAM_SIZE

# ── DOWN path: label → x1 → x0 → image ────────────────────────────────────
comptime DB0 = PCBlock[10, HIDDEN, PCIdentity]
comptime DB1 = PCBlock[HIDDEN, HIDDEN, PCReLU]
comptime DB2 = PCBlock[HIDDEN, 784, PCReLU]
comptime DOWN_NET = PCSequential[DB0, DB1, DB2]
comptime DOWN_PARAM_SIZE = DOWN_NET.PARAM_SIZE

comptime OPT = PCAdam[LR=ADAM_LR]


def main() raises:
    print("=" * 60)
    print("Bidirectional PC — Bogacz notebook 5 (full params)")
    print("=" * 60)
    print("  UP   arch  : 784 →", HIDDEN, "→", HIDDEN, "→ 10")
    print("  DOWN arch  : 10  →", HIDDEN, "→", HIDDEN, "→ 784")
    print("  UP params  :", UP_PARAM_SIZE, " DOWN params:", DOWN_PARAM_SIZE)
    print("  hyperparams: BATCH=", BATCH, " T_INFER=", T_INFER, " EPOCHS=", EPOCHS)
    print("  α_up=", ALPHA_UP, " α_down=", ALPHA_DOWN)

    var ds = MNIST()
    print("  [mnist] loaded:", MNIST.N_TRAIN, "train,", MNIST.N_TEST, "test")

    # ── Allocate UP params + Adam state ─────────────────────────────────────
    var up_params_buf = alloc[Scalar[dtype]](UP_PARAM_SIZE).as_unsafe_any_origin()
    var up_grads_buf = alloc[Scalar[dtype]](UP_PARAM_SIZE).as_unsafe_any_origin()
    var up_opt_state_buf = alloc[Scalar[dtype]](UP_PARAM_SIZE * OPT.STATE_PER_PARAM).as_unsafe_any_origin()
    var up_opt_global_buf = alloc[Scalar[dtype]](OPT.GLOBAL_STATE_SIZE).as_unsafe_any_origin()
    memset(up_params_buf, 0, UP_PARAM_SIZE)
    memset(up_grads_buf, 0, UP_PARAM_SIZE)
    memset(up_opt_state_buf, 0, UP_PARAM_SIZE * OPT.STATE_PER_PARAM)
    memset(up_opt_global_buf, 0, OPT.GLOBAL_STATE_SIZE)
    var up_params = tile[UP_PARAM_SIZE](up_params_buf)
    var up_grads = tile[UP_PARAM_SIZE](up_grads_buf)
    var up_opt_state = tile[UP_PARAM_SIZE, OPT.STATE_PER_PARAM](up_opt_state_buf)
    var up_opt_global = tile[OPT.GLOBAL_STATE_SIZE](up_opt_global_buf)
    UP_NET.pc_init_params[PCXavier, dtype](up_params)

    # ── Allocate DOWN params + Adam state ───────────────────────────────────
    var dn_params_buf = alloc[Scalar[dtype]](DOWN_PARAM_SIZE).as_unsafe_any_origin()
    var dn_grads_buf = alloc[Scalar[dtype]](DOWN_PARAM_SIZE).as_unsafe_any_origin()
    var dn_opt_state_buf = alloc[Scalar[dtype]](DOWN_PARAM_SIZE * OPT.STATE_PER_PARAM).as_unsafe_any_origin()
    var dn_opt_global_buf = alloc[Scalar[dtype]](OPT.GLOBAL_STATE_SIZE).as_unsafe_any_origin()
    memset(dn_params_buf, 0, DOWN_PARAM_SIZE)
    memset(dn_grads_buf, 0, DOWN_PARAM_SIZE)
    memset(dn_opt_state_buf, 0, DOWN_PARAM_SIZE * OPT.STATE_PER_PARAM)
    memset(dn_opt_global_buf, 0, OPT.GLOBAL_STATE_SIZE)
    var dn_params = tile[DOWN_PARAM_SIZE](dn_params_buf)
    var dn_grads = tile[DOWN_PARAM_SIZE](dn_grads_buf)
    var dn_opt_state = tile[DOWN_PARAM_SIZE, OPT.STATE_PER_PARAM](dn_opt_state_buf)
    var dn_opt_global = tile[OPT.GLOBAL_STATE_SIZE](dn_opt_global_buf)
    DOWN_NET.pc_init_params[PCXavier, dtype](dn_params)

    # ── Shared latents x0, x1 (both HIDDEN-dim) ────────────────────────────
    var x0_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var x1_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    memset(x0_buf, 0, BATCH * HIDDEN)
    memset(x1_buf, 0, BATCH * HIDDEN)
    var x0 = tile[BATCH, HIDDEN](x0_buf)
    var x1 = tile[BATCH, HIDDEN](x1_buf)

    # ── Per-block scratch buffers ───────────────────────────────────────────
    # UP: block_0 (784→256), block_1 (256→256), block_2 (256→10)
    var up_mu0_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var up_eps0_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var up_a0_buf = alloc[Scalar[dtype]](BATCH * 784).as_unsafe_any_origin()
    var up_mu1_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var up_eps1_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var up_a1_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var up_mu2_buf = alloc[Scalar[dtype]](BATCH * 10).as_unsafe_any_origin()
    var up_eps2_buf = alloc[Scalar[dtype]](BATCH * 10).as_unsafe_any_origin()
    var up_a2_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var up_z1_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()  # pull_back ε_up_1 → for x0
    var up_z2_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()  # pull_back ε_up_2 → for x1

    # DOWN: block_0 (10→256), block_1 (256→256), block_2 (256→784)
    var dn_mu0_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var dn_eps0_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var dn_a0_buf = alloc[Scalar[dtype]](BATCH * 10).as_unsafe_any_origin()
    var dn_mu1_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var dn_eps1_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var dn_a1_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var dn_mu2_buf = alloc[Scalar[dtype]](BATCH * 784).as_unsafe_any_origin()
    var dn_eps2_buf = alloc[Scalar[dtype]](BATCH * 784).as_unsafe_any_origin()
    var dn_a2_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var dn_z1_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()  # pull_back ε_dn_1 → for x1
    var dn_z2_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()  # pull_back ε_dn_2 → for x0

    var dx0_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var dx1_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()

    # ── Per-batch input buffers ─────────────────────────────────────────────
    var image_buf = alloc[Scalar[dtype]](BATCH * 784).as_unsafe_any_origin()
    var label_oh_buf = alloc[Scalar[dtype]](BATCH * 10).as_unsafe_any_origin()
    memset(image_buf, 0, BATCH * 784)
    memset(label_oh_buf, 0, BATCH * 10)

    # ── Param block views ───────────────────────────────────────────────────
    var up_p0 = tile[UB0.PARAM_SIZE](up_params_buf)
    var up_p1 = tile[UB1.PARAM_SIZE](up_params_buf + UB0.PARAM_SIZE)
    var up_p2 = tile[UB2.PARAM_SIZE](up_params_buf + UB0.PARAM_SIZE + UB1.PARAM_SIZE)
    var dn_p0 = tile[DB0.PARAM_SIZE](dn_params_buf)
    var dn_p1 = tile[DB1.PARAM_SIZE](dn_params_buf + DB0.PARAM_SIZE)
    var dn_p2 = tile[DB2.PARAM_SIZE](dn_params_buf + DB0.PARAM_SIZE + DB1.PARAM_SIZE)
    var up_g0 = tile[UB0.PARAM_SIZE](up_grads_buf)
    var up_g1 = tile[UB1.PARAM_SIZE](up_grads_buf + UB0.PARAM_SIZE)
    var up_g2 = tile[UB2.PARAM_SIZE](up_grads_buf + UB0.PARAM_SIZE + UB1.PARAM_SIZE)
    var dn_g0 = tile[DB0.PARAM_SIZE](dn_grads_buf)
    var dn_g1 = tile[DB1.PARAM_SIZE](dn_grads_buf + DB0.PARAM_SIZE)
    var dn_g2 = tile[DB2.PARAM_SIZE](dn_grads_buf + DB0.PARAM_SIZE + DB1.PARAM_SIZE)

    # ── Tensor views ────────────────────────────────────────────────────────
    var image_t = tile[BATCH, 784](image_buf)
    var label_oh_t = tile[BATCH, 10](label_oh_buf)
    var up_mu0_t = tile[BATCH, HIDDEN](up_mu0_buf)
    var up_eps0_t = tile[BATCH, HIDDEN](up_eps0_buf)
    var up_a0_t = tile[BATCH, 784](up_a0_buf)
    var up_mu1_t = tile[BATCH, HIDDEN](up_mu1_buf)
    var up_eps1_t = tile[BATCH, HIDDEN](up_eps1_buf)
    var up_a1_t = tile[BATCH, HIDDEN](up_a1_buf)
    var up_mu2_t = tile[BATCH, 10](up_mu2_buf)
    var up_eps2_t = tile[BATCH, 10](up_eps2_buf)
    var up_a2_t = tile[BATCH, HIDDEN](up_a2_buf)
    var up_z1_t = tile[BATCH, HIDDEN](up_z1_buf)
    var up_z2_t = tile[BATCH, HIDDEN](up_z2_buf)
    var dn_mu0_t = tile[BATCH, HIDDEN](dn_mu0_buf)
    var dn_eps0_t = tile[BATCH, HIDDEN](dn_eps0_buf)
    var dn_a0_t = tile[BATCH, 10](dn_a0_buf)
    var dn_mu1_t = tile[BATCH, HIDDEN](dn_mu1_buf)
    var dn_eps1_t = tile[BATCH, HIDDEN](dn_eps1_buf)
    var dn_a1_t = tile[BATCH, HIDDEN](dn_a1_buf)
    var dn_mu2_t = tile[BATCH, 784](dn_mu2_buf)
    var dn_eps2_t = tile[BATCH, 784](dn_eps2_buf)
    var dn_a2_t = tile[BATCH, HIDDEN](dn_a2_buf)
    var dn_z1_t = tile[BATCH, HIDDEN](dn_z1_buf)
    var dn_z2_t = tile[BATCH, HIDDEN](dn_z2_buf)

    # ── Train ───────────────────────────────────────────────────────────────
    print("\n  epoch | up_loss | dn_loss | test_acc | wall_t (s)")
    print("  ------+---------+---------+----------+------------")
    var step_num: Int = 0
    var t0 = perf_counter_ns()

    for epoch in range(EPOCHS):
        var ep_up_loss: Float64 = 0.0
        var ep_dn_loss: Float64 = 0.0
        for batch_idx in range(N_TRAIN_BATCHES):
            # Load batch
            for i in range(BATCH):
                var sample_idx = batch_idx * BATCH + i
                for j in range(784):
                    image_buf[i * 784 + j] = ds.train_images[sample_idx * 784 + j]
                for c in range(10):
                    label_oh_buf[i * 10 + c] = Scalar[dtype](0)
                label_oh_buf[i * 10 + Int(ds.train_labels[sample_idx])] = Scalar[dtype](1.0)

            # ── Init shared latents via UP forward: x0 = μ_up_0, x1 = μ_up_1 ──
            UB0.predict[BATCH, dtype](image_t, up_p0, up_mu0_t, up_a0_t)
            for i in range(BATCH * HIDDEN):
                x0_buf[i] = up_mu0_buf[i]
            UB1.predict[BATCH, dtype](x0, up_p1, up_mu1_t, up_a1_t)
            for i in range(BATCH * HIDDEN):
                x1_buf[i] = up_mu1_buf[i]

            # ── T_INFER iterations of joint inference ──
            for _ in range(T_INFER):
                # UP predictions
                UB0.predict[BATCH, dtype](image_t, up_p0, up_mu0_t, up_a0_t)    # μ_up_0 from image → predicts x0
                UB1.predict[BATCH, dtype](x0, up_p1, up_mu1_t, up_a1_t)         # μ_up_1 from x0 → predicts x1
                UB2.predict[BATCH, dtype](x1, up_p2, up_mu2_t, up_a2_t)         # μ_up_2 from x1 → predicts label
                UB0.eps_compute[BATCH, dtype](x0, up_mu0_t, up_eps0_t)          # ε_up_0 = x0 - μ_up_0
                UB1.eps_compute[BATCH, dtype](x1, up_mu1_t, up_eps1_t)          # ε_up_1 = x1 - μ_up_1
                UB2.eps_compute[BATCH, dtype](label_oh_t, up_mu2_t, up_eps2_t)  # ε_up_2 = label - μ_up_2

                # DOWN predictions
                DB0.predict[BATCH, dtype](label_oh_t, dn_p0, dn_mu0_t, dn_a0_t) # μ_dn_0 from label → predicts x1
                DB1.predict[BATCH, dtype](x1, dn_p1, dn_mu1_t, dn_a1_t)         # μ_dn_1 from x1 → predicts x0
                DB2.predict[BATCH, dtype](x0, dn_p2, dn_mu2_t, dn_a2_t)         # μ_dn_2 from x0 → predicts image
                DB0.eps_compute[BATCH, dtype](x1, dn_mu0_t, dn_eps0_t)          # ε_dn_0 = x1 - μ_dn_0
                DB1.eps_compute[BATCH, dtype](x0, dn_mu1_t, dn_eps1_t)          # ε_dn_1 = x0 - μ_dn_1
                DB2.eps_compute[BATCH, dtype](image_t, dn_mu2_t, dn_eps2_t)     # ε_dn_2 = image - μ_dn_2

                # Pull-backs for x0 gradient:
                #   UP:  z_up_1  = act'(x0) ⊙ W_up_1ᵀ · ε_up_1
                #   DOWN: z_dn_2 = act'(x0) ⊙ W_dn_2ᵀ · ε_dn_2
                UB1.pull_back[BATCH, dtype](up_eps1_t, up_p1, up_z1_t)
                UB1.act_derivative_mul[BATCH, dtype](x0, up_z1_t, up_z1_t)
                DB2.pull_back[BATCH, dtype](dn_eps2_t, dn_p2, dn_z2_t)
                DB2.act_derivative_mul[BATCH, dtype](x0, dn_z2_t, dn_z2_t)

                # Pull-backs for x1 gradient:
                #   UP:  z_up_2  = act'(x1) ⊙ W_up_2ᵀ · ε_up_2
                #   DOWN: z_dn_1 = act'(x1) ⊙ W_dn_1ᵀ · ε_dn_1
                UB2.pull_back[BATCH, dtype](up_eps2_t, up_p2, up_z2_t)
                UB2.act_derivative_mul[BATCH, dtype](x1, up_z2_t, up_z2_t)
                DB1.pull_back[BATCH, dtype](dn_eps1_t, dn_p1, dn_z1_t)
                DB1.act_derivative_mul[BATCH, dtype](x1, dn_z1_t, dn_z1_t)

                # dx0 = α_up·(ε_up_0 − z_up_1) + α_down·(ε_dn_1 − z_dn_2)
                # dx1 = α_up·(ε_up_1 − z_up_2) + α_down·(ε_dn_0 − z_dn_1)
                for i in range(BATCH * HIDDEN):
                    var u0 = Float64(up_eps0_buf[i]) - Float64(up_z1_buf[i])
                    var d0 = Float64(dn_eps1_buf[i]) - Float64(dn_z2_buf[i])
                    dx0_buf[i] = Scalar[dtype](ALPHA_UP * u0 + ALPHA_DOWN * d0)
                    var u1 = Float64(up_eps1_buf[i]) - Float64(up_z2_buf[i])
                    var d1 = Float64(dn_eps0_buf[i]) - Float64(dn_z1_buf[i])
                    dx1_buf[i] = Scalar[dtype](ALPHA_UP * u1 + ALPHA_DOWN * d1)

                # SGD update on shared latents
                for i in range(BATCH * HIDDEN):
                    x0_buf[i] = x0_buf[i] - Scalar[dtype](LR_X) * dx0_buf[i]
                    x1_buf[i] = x1_buf[i] - Scalar[dtype](LR_X) * dx1_buf[i]

            # ── Weight grads (post-inference ε) ─────────────────────────────
            UB0.weight_grad[BATCH, dtype](up_eps0_t, up_a0_t, up_g0)
            UB1.weight_grad[BATCH, dtype](up_eps1_t, up_a1_t, up_g1)
            UB2.weight_grad[BATCH, dtype](up_eps2_t, up_a2_t, up_g2)
            DB0.weight_grad[BATCH, dtype](dn_eps0_t, dn_a0_t, dn_g0)
            DB1.weight_grad[BATCH, dtype](dn_eps1_t, dn_a1_t, dn_g1)
            DB2.weight_grad[BATCH, dtype](dn_eps2_t, dn_a2_t, dn_g2)

            # Scale DOWN grads by alpha_down
            for i in range(DOWN_PARAM_SIZE):
                dn_grads_buf[i] = dn_grads_buf[i] * Scalar[dtype](ALPHA_DOWN)

            # Adam steps
            step_num += 1
            OPT.step[UP_PARAM_SIZE, dtype](up_params, up_grads, up_opt_state, up_opt_global, step_num)
            OPT.step[DOWN_PARAM_SIZE, dtype](dn_params, dn_grads, dn_opt_state, dn_opt_global, step_num)

            # Track losses
            var sup_loss: Float64 = 0
            for i in range(BATCH * 10):
                var v = Float64(up_eps2_buf[i])
                sup_loss += v * v
            ep_up_loss += sup_loss * 0.5
            var rec_loss: Float64 = 0
            for i in range(BATCH * 784):
                var v = Float64(dn_eps2_buf[i])
                rec_loss += v * v
            ep_dn_loss += rec_loss * 0.5

        # End-of-epoch: quick accuracy check
        var correct: Int = 0
        var pred_buf = alloc[Scalar[dtype]](BATCH * 10).as_unsafe_any_origin()
        memset(pred_buf, 0, BATCH * 10)
        var pred_t = tile[BATCH, 10](pred_buf)
        for tb in range(N_TEST_BATCHES):
            for i in range(BATCH):
                var sample_idx = tb * BATCH + i
                for j in range(784):
                    image_buf[i * 784 + j] = ds.test_images[sample_idx * 784 + j]
            UP_NET.forward_eval[BATCH, dtype](image_t, up_params, pred_t)
            for i in range(BATCH):
                var best_class: Int = 0
                var best_val = Float64(pred_buf[i * 10])
                for c in range(1, 10):
                    var v = Float64(pred_buf[i * 10 + c])
                    if v > best_val:
                        best_val = v
                        best_class = c
                var sample_idx = tb * BATCH + i
                if best_class == Int(ds.test_labels[sample_idx]):
                    correct += 1
        var test_acc = Float64(correct) / Float64(N_TEST_BATCHES * BATCH)
        pred_buf.free()

        var elapsed = Float64(perf_counter_ns() - t0) / 1e9
        var avg_up = ep_up_loss / Float64(N_TRAIN_BATCHES)
        var avg_dn = ep_dn_loss / Float64(N_TRAIN_BATCHES)
        print("    ", epoch, "  ", avg_up, "  ", avg_dn, "  ", test_acc, "  ", elapsed)

    var total_t = Float64(perf_counter_ns() - t0) / 1e9
    print("\n  total train time:", total_t, "s")

    # ── Generation: pass each one-hot label through DOWN forward_eval ───────
    var gen_label_buf = alloc[Scalar[dtype]](BATCH * 10).as_unsafe_any_origin()
    var gen_image_buf = alloc[Scalar[dtype]](BATCH * 784).as_unsafe_any_origin()
    memset(gen_label_buf, 0, BATCH * 10)
    memset(gen_image_buf, 0, BATCH * 784)
    for c in range(10):
        gen_label_buf[c * 10 + c] = Scalar[dtype](1.0)
    var gen_label_t = tile[BATCH, 10](gen_label_buf)
    var gen_image_t = tile[BATCH, 784](gen_image_buf)
    DOWN_NET.forward_eval[BATCH, dtype](gen_label_t, dn_params, gen_image_t)

    # Stats on generated images
    var class_diff: Float64 = 0
    for j in range(784):
        var d = Float64(gen_image_buf[0 * 784 + j]) - Float64(gen_image_buf[9 * 784 + j])
        class_diff += d * d
    class_diff /= Float64(784)
    print("  Per-pixel MSE between class-0 and class-9:", class_diff)

    var gen_mean_mag: Float64 = 0
    for c in range(10):
        for j in range(784):
            gen_mean_mag += abs(Float64(gen_image_buf[c * 784 + j]))
    gen_mean_mag /= Float64(10 * 784)
    print("  Mean |gen pixel| across 10 classes:", gen_mean_mag)

    # ── Visualize ───────────────────────────────────────────────────────────
    # Find one real test sample per class for comparison
    var real_digits_buf = alloc[Scalar[dtype]](10 * 784).as_unsafe_any_origin()
    var found = alloc[UInt8](10).as_unsafe_any_origin()
    memset(found, 0, 10)
    var found_count: Int = 0
    for i in range(MNIST.N_TEST):
        var c = Int(ds.test_labels[i])
        if found[c] == 0:
            for j in range(784):
                real_digits_buf[c * 784 + j] = ds.test_images[i * 784 + j]
            found[c] = 1
            found_count += 1
            if found_count == 10:
                break
    found.free()

    # Apply sigmoid to generated images (matches notebook: img.sigmoid())
    var gen_sig_buf = alloc[Scalar[dtype]](10 * 784).as_unsafe_any_origin()
    for i in range(10 * 784):
        var v = Float64(gen_image_buf[i])
        gen_sig_buf[i] = Scalar[dtype](1.0 / (1.0 + exp(-v)))
    # After sigmoid, values are in [0, 1]

    var digit_labels = List[String]()
    for i in range(10):
        digit_labels.append(String(i))

    save_image_row(
        "pcn_generated_digits.ppm", gen_sig_buf,
        n=10, height=28, width=28, channels=1,
        vmin=0.0, vmax=1.0, pixel_scale=4, labels=digit_labels,
    )
    save_image_row(
        "pcn_real_digits.ppm", real_digits_buf,
        n=10, height=28, width=28, channels=1,
        vmin=0.0, vmax=1.0, pixel_scale=4, labels=digit_labels,
    )
    save_reconstruction_grid(
        "pcn_real_vs_generated.ppm", real_digits_buf, gen_sig_buf,
        n=10, height=28, width=28, channels=1,
        vmin=0.0, vmax=1.0,
    )
    print("  Visualizations saved — open with: open pcn_generated_digits.ppm pcn_real_vs_generated.ppm pcn_real_digits.ppm")

    gen_sig_buf.free()
    real_digits_buf.free()

    # ── Verdict ─────────────────────────────────────────────────────────────
    # Re-run final accuracy (already computed above but recompute clean)
    var final_pred_buf = alloc[Scalar[dtype]](BATCH * 10).as_unsafe_any_origin()
    memset(final_pred_buf, 0, BATCH * 10)
    var final_pred_t = tile[BATCH, 10](final_pred_buf)
    var final_correct: Int = 0
    for tb in range(N_TEST_BATCHES):
        for i in range(BATCH):
            var sample_idx = tb * BATCH + i
            for j in range(784):
                image_buf[i * 784 + j] = ds.test_images[sample_idx * 784 + j]
        UP_NET.forward_eval[BATCH, dtype](image_t, up_params, final_pred_t)
        for i in range(BATCH):
            var best_class: Int = 0
            var best_val = Float64(final_pred_buf[i * 10])
            for c in range(1, 10):
                var v = Float64(final_pred_buf[i * 10 + c])
                if v > best_val:
                    best_val = v
                    best_class = c
            var sample_idx = tb * BATCH + i
            if best_class == Int(ds.test_labels[sample_idx]):
                final_correct += 1
    var final_acc = Float64(final_correct) / Float64(N_TEST_BATCHES * BATCH)
    final_pred_buf.free()

    print("\n  Final test accuracy:", final_acc)
    var pass_acc = final_acc >= 0.70
    print("  classification check (≥70%):", pass_acc, " (got ", final_acc, ")")
    print("  generation diversity (info) : per-class MSE =", class_diff)
    if pass_acc:
        print("\n  [PASS] bidirectional PC (notebook 5 params)")
    else:
        print("\n  [FAIL] bidirectional PC: accuracy too low")
        raise Error("bidirectional PC test failed")

    # ── Cleanup ─────────────────────────────────────────────────────────────
    up_params_buf.free()
    up_grads_buf.free()
    up_opt_state_buf.free()
    up_opt_global_buf.free()
    dn_params_buf.free()
    dn_grads_buf.free()
    dn_opt_state_buf.free()
    dn_opt_global_buf.free()
    x0_buf.free()
    x1_buf.free()
    up_mu0_buf.free()
    up_eps0_buf.free()
    up_a0_buf.free()
    up_mu1_buf.free()
    up_eps1_buf.free()
    up_a1_buf.free()
    up_mu2_buf.free()
    up_eps2_buf.free()
    up_a2_buf.free()
    up_z1_buf.free()
    up_z2_buf.free()
    dn_mu0_buf.free()
    dn_eps0_buf.free()
    dn_a0_buf.free()
    dn_mu1_buf.free()
    dn_eps1_buf.free()
    dn_a1_buf.free()
    dn_mu2_buf.free()
    dn_eps2_buf.free()
    dn_a2_buf.free()
    dn_z1_buf.free()
    dn_z2_buf.free()
    dx0_buf.free()
    dx1_buf.free()
    image_buf.free()
    label_oh_buf.free()
    gen_label_buf.free()
    gen_image_buf.free()
    print("=== Done ===")


