"""Bidirectional CONV PC — conv recognition + transposed-conv generation (CPU).

Conv adaptation of test_bidirectional_pc.mojo (Bogacz notebook 5). Two paths
share latents x0 (8×14×14) and x1 (16×7×7):
  - UP   (recognize): image 1×28×28 →[conv s2] x0 →[conv s2] x1 →[MLP] label
  - DOWN (generate) : label →[MLP] x1 →[convTranspose s2] x0 →[convTranspose s2] image

The generative path produces images with a real conv decoder (bottleneck
7×7 → 14×14 → 28×28), so generated per-class digits reflect local conv
structure rather than a flat MLP decoder.

Total energy = α_up·E_up + α_down·E_down; shared latents updated by the SUM of
both paths' gradients per inference step.

Outputs: pcn_conv_generated_digits.ppm, pcn_conv_real_vs_generated.ppm.

Run:
    pixi run mojo run -I . tests/pcn/test_bidirectional_conv_pc.mojo
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
from noeira.experimental.pcn.pc_conv_block import ConvPCBlock
from noeira.experimental.pcn.pc_conv_transpose_block import (
    ConvTransposePCBlock,
)
from noeira.render.image_writer import save_image_row, save_reconstruction_grid

# ── Hyperparameters (trimmed from notebook 5 for CPU budget) ────────────────
comptime BATCH = 200
comptime EPOCHS = 15
comptime T_INFER = 12
comptime LR_X: Float64 = 0.01
comptime ADAM_LR: Float64 = 0.01
comptime ALPHA_UP: Float64 = 1.0
comptime ALPHA_DOWN: Float64 = 0.03  # stronger reconstruction → sharper gen
comptime N_TRAIN = 6000
comptime N_TEST = 1000
comptime N_TRAIN_BATCHES = N_TRAIN // BATCH
comptime N_TEST_BATCHES = N_TEST // BATCH

# ── Channels / latent dims ──────────────────────────────────────────────────
comptime C0 = 16
comptime C1 = 32
comptime X0_DIM = C0 * 14 * 14  # 1568
comptime X1_DIM = C1 * 7 * 7  # 784
comptime IMG = 784
comptime LBL = 10

# ── UP path: image → x0 → x1 → label ──────────────────────────────────────
comptime UB0 = ConvPCBlock[1, C0, 4, 2, 1, 28, 28, PCIdentity]  # 784 → 1568
comptime UB1 = ConvPCBlock[C0, C1, 4, 2, 1, 14, 14, PCReLU]  # 1568 → 784
comptime UB2 = PCBlock[X1_DIM, 10, PCReLU]  # 784 → 10
comptime UP_NET = PCSequential[UB0, UB1, UB2]
comptime UP_PARAM_SIZE = UP_NET.PARAM_SIZE

# ── DOWN path: label → x1 → x0 → image ────────────────────────────────────
comptime DB0 = PCBlock[10, X1_DIM, PCIdentity]  # 10 → 784
comptime DB1 = ConvTransposePCBlock[C1, C0, 4, 2, 1, 7, 7, PCReLU]  # 784 → 1568
comptime DB2 = ConvTransposePCBlock[
    C0, 1, 4, 2, 1, 14, 14, PCReLU
]  # 1568 → 784
comptime DOWN_PARAM_SIZE = DB0.PARAM_SIZE + DB1.PARAM_SIZE + DB2.PARAM_SIZE

comptime OPT = PCAdam[LR=ADAM_LR]


def main() raises:
    print("=" * 60)
    print("Bidirectional CONV PC (conv recognize + convT generate)")
    print("=" * 60)
    print("  UP  : image 1x28x28 → x0", C0, "x14x14 → x1", C1, "x7x7 → label")
    print("  DOWN: label → x1 → x0 → image (transposed conv decoder)")
    print("  X0_DIM=", X0_DIM, " X1_DIM=", X1_DIM)
    print("  UP params=", UP_PARAM_SIZE, " DOWN params=", DOWN_PARAM_SIZE)
    print("  BATCH=", BATCH, " T_INFER=", T_INFER, " EPOCHS=", EPOCHS)

    var ds = MNIST()

    # ── UP params + Adam ────────────────────────────────────────────────────
    var up_params_buf = alloc[Scalar[dtype]](UP_PARAM_SIZE).as_unsafe_any_origin()
    var up_grads_buf = alloc[Scalar[dtype]](UP_PARAM_SIZE).as_unsafe_any_origin()
    var up_os_buf = alloc[Scalar[dtype]](UP_PARAM_SIZE * OPT.STATE_PER_PARAM).as_unsafe_any_origin()
    var up_og_buf = alloc[Scalar[dtype]](OPT.GLOBAL_STATE_SIZE).as_unsafe_any_origin()
    memset(up_params_buf, 0, UP_PARAM_SIZE)
    memset(up_grads_buf, 0, UP_PARAM_SIZE)
    memset(up_os_buf, 0, UP_PARAM_SIZE * OPT.STATE_PER_PARAM)
    memset(up_og_buf, 0, OPT.GLOBAL_STATE_SIZE)
    var up_params = tile[UP_PARAM_SIZE](up_params_buf)
    var up_grads = tile[UP_PARAM_SIZE](up_grads_buf)
    var up_os = tile[UP_PARAM_SIZE, OPT.STATE_PER_PARAM](up_os_buf)
    var up_og = tile[OPT.GLOBAL_STATE_SIZE](up_og_buf)
    UP_NET.pc_init_params[PCXavier, dtype](up_params)

    # ── DOWN params + Adam (init each block separately) ─────────────────────
    var dn_params_buf = alloc[Scalar[dtype]](DOWN_PARAM_SIZE).as_unsafe_any_origin()
    var dn_grads_buf = alloc[Scalar[dtype]](DOWN_PARAM_SIZE).as_unsafe_any_origin()
    var dn_os_buf = alloc[Scalar[dtype]](DOWN_PARAM_SIZE * OPT.STATE_PER_PARAM).as_unsafe_any_origin()
    var dn_og_buf = alloc[Scalar[dtype]](OPT.GLOBAL_STATE_SIZE).as_unsafe_any_origin()
    memset(dn_params_buf, 0, DOWN_PARAM_SIZE)
    memset(dn_grads_buf, 0, DOWN_PARAM_SIZE)
    memset(dn_os_buf, 0, DOWN_PARAM_SIZE * OPT.STATE_PER_PARAM)
    memset(dn_og_buf, 0, OPT.GLOBAL_STATE_SIZE)
    var dn_params = tile[DOWN_PARAM_SIZE](dn_params_buf)
    var dn_grads = tile[DOWN_PARAM_SIZE](dn_grads_buf)
    var dn_os = tile[DOWN_PARAM_SIZE, OPT.STATE_PER_PARAM](dn_os_buf)
    var dn_og = tile[OPT.GLOBAL_STATE_SIZE](dn_og_buf)
    var dn_p0v = tile[DB0.PARAM_SIZE](dn_params_buf)
    var dn_p1v = tile[DB1.PARAM_SIZE](dn_params_buf + DB0.PARAM_SIZE)
    var dn_p2v = tile[DB2.PARAM_SIZE](dn_params_buf + DB0.PARAM_SIZE + DB1.PARAM_SIZE)
    DB0.pc_init_params[PCXavier, dtype](dn_p0v)
    DB1.pc_init_params[PCXavier, dtype](dn_p1v)
    DB2.pc_init_params[PCXavier, dtype](dn_p2v)

    # ── Shared latents ──────────────────────────────────────────────────────
    var x0_buf = alloc[Scalar[dtype]](BATCH * X0_DIM).as_unsafe_any_origin()
    var x1_buf = alloc[Scalar[dtype]](BATCH * X1_DIM).as_unsafe_any_origin()
    memset(x0_buf, 0, BATCH * X0_DIM)
    memset(x1_buf, 0, BATCH * X1_DIM)
    var x0 = tile[BATCH, X0_DIM](x0_buf)
    var x1 = tile[BATCH, X1_DIM](x1_buf)

    # ── Scratch (UP) ────────────────────────────────────────────────────────
    var up_mu0 = alloc[Scalar[dtype]](BATCH * X0_DIM).as_unsafe_any_origin()
    var up_eps0 = alloc[Scalar[dtype]](BATCH * X0_DIM).as_unsafe_any_origin()
    var up_a0 = alloc[Scalar[dtype]](BATCH * IMG).as_unsafe_any_origin()
    var up_mu1 = alloc[Scalar[dtype]](BATCH * X1_DIM).as_unsafe_any_origin()
    var up_eps1 = alloc[Scalar[dtype]](BATCH * X1_DIM).as_unsafe_any_origin()
    var up_a1 = alloc[Scalar[dtype]](BATCH * X0_DIM).as_unsafe_any_origin()
    var up_mu2 = alloc[Scalar[dtype]](BATCH * 10).as_unsafe_any_origin()
    var up_eps2 = alloc[Scalar[dtype]](BATCH * 10).as_unsafe_any_origin()
    var up_a2 = alloc[Scalar[dtype]](BATCH * X1_DIM).as_unsafe_any_origin()
    var up_z1 = alloc[Scalar[dtype]](BATCH * X0_DIM).as_unsafe_any_origin()  # pull_back ε_up1 → x0
    var up_z2 = alloc[Scalar[dtype]](BATCH * X1_DIM).as_unsafe_any_origin()  # pull_back ε_up2 → x1

    # ── Scratch (DOWN) ──────────────────────────────────────────────────────
    var dn_mu0 = alloc[Scalar[dtype]](BATCH * X1_DIM).as_unsafe_any_origin()
    var dn_eps0 = alloc[Scalar[dtype]](BATCH * X1_DIM).as_unsafe_any_origin()
    var dn_a0 = alloc[Scalar[dtype]](BATCH * 10).as_unsafe_any_origin()
    var dn_mu1 = alloc[Scalar[dtype]](BATCH * X0_DIM).as_unsafe_any_origin()
    var dn_eps1 = alloc[Scalar[dtype]](BATCH * X0_DIM).as_unsafe_any_origin()
    var dn_a1 = alloc[Scalar[dtype]](BATCH * X1_DIM).as_unsafe_any_origin()
    var dn_mu2 = alloc[Scalar[dtype]](BATCH * IMG).as_unsafe_any_origin()
    var dn_eps2 = alloc[Scalar[dtype]](BATCH * IMG).as_unsafe_any_origin()
    var dn_a2 = alloc[Scalar[dtype]](BATCH * X0_DIM).as_unsafe_any_origin()
    var dn_z1 = alloc[Scalar[dtype]](BATCH * X1_DIM).as_unsafe_any_origin()  # pull_back ε_dn1 → x1
    var dn_z2 = alloc[Scalar[dtype]](BATCH * X0_DIM).as_unsafe_any_origin()  # pull_back ε_dn2 → x0

    var dx0 = alloc[Scalar[dtype]](BATCH * X0_DIM).as_unsafe_any_origin()
    var dx1 = alloc[Scalar[dtype]](BATCH * X1_DIM).as_unsafe_any_origin()

    var image_buf = alloc[Scalar[dtype]](BATCH * IMG).as_unsafe_any_origin()
    var label_buf = alloc[Scalar[dtype]](BATCH * 10).as_unsafe_any_origin()
    memset(image_buf, 0, BATCH * IMG)
    memset(label_buf, 0, BATCH * 10)

    # ── UP param block views ────────────────────────────────────────────────
    var up_p0 = tile[UB0.PARAM_SIZE](up_params_buf)
    var up_p1 = tile[UB1.PARAM_SIZE](up_params_buf + UB0.PARAM_SIZE)
    var up_p2 = tile[UB2.PARAM_SIZE](up_params_buf + UB0.PARAM_SIZE + UB1.PARAM_SIZE)
    var up_g0 = tile[UB0.PARAM_SIZE](up_grads_buf)
    var up_g1 = tile[UB1.PARAM_SIZE](up_grads_buf + UB0.PARAM_SIZE)
    var up_g2 = tile[UB2.PARAM_SIZE](up_grads_buf + UB0.PARAM_SIZE + UB1.PARAM_SIZE)
    var dn_g0 = tile[DB0.PARAM_SIZE](dn_grads_buf)
    var dn_g1 = tile[DB1.PARAM_SIZE](dn_grads_buf + DB0.PARAM_SIZE)
    var dn_g2 = tile[DB2.PARAM_SIZE](dn_grads_buf + DB0.PARAM_SIZE + DB1.PARAM_SIZE)

    # ── Tensor views (typed by the CONSUMING block's member — conv blocks'
    #    computed dims don't unify with literal aliases, so shared latents get
    #    one view per role) ──────────────────────────────────────────────────
    # image: UB0 input (recognize) + DB2 output (reconstruct)
    var image_ub0 = tile[BATCH, UB0.IN_DIM](image_buf)
    var image_db2 = tile[BATCH, DB2.OUT_DIM](image_buf)
    # label: UB2 output (classify) + DB0 input (generate)
    var label_ub2 = tile[BATCH, UB2.OUT_DIM](label_buf)
    var label_db0 = tile[BATCH, DB0.IN_DIM](label_buf)
    # x0: UB0 above, UB1 below, DB1 above, DB2 below
    var x0_ub0 = tile[BATCH, UB0.OUT_DIM](x0_buf)
    var x0_ub1 = tile[BATCH, UB1.IN_DIM](x0_buf)
    var x0_db1 = tile[BATCH, DB1.OUT_DIM](x0_buf)
    var x0_db2 = tile[BATCH, DB2.IN_DIM](x0_buf)
    # x1: UB1 above, UB2 below, DB0 above, DB1 below
    var x1_ub1 = tile[BATCH, UB1.OUT_DIM](x1_buf)
    var x1_ub2 = tile[BATCH, UB2.IN_DIM](x1_buf)
    var x1_db0 = tile[BATCH, DB0.OUT_DIM](x1_buf)
    var x1_db1 = tile[BATCH, DB1.IN_DIM](x1_buf)
    # scratch (owner-block typed)
    var up_mu0_t = tile[BATCH, UB0.OUT_DIM](up_mu0)
    var up_eps0_t = tile[BATCH, UB0.OUT_DIM](up_eps0)
    var up_a0_t = tile[BATCH, UB0.IN_DIM](up_a0)
    var up_mu1_t = tile[BATCH, UB1.OUT_DIM](up_mu1)
    var up_eps1_t = tile[BATCH, UB1.OUT_DIM](up_eps1)
    var up_a1_t = tile[BATCH, UB1.IN_DIM](up_a1)
    var up_mu2_t = tile[BATCH, UB2.OUT_DIM](up_mu2)
    var up_eps2_t = tile[BATCH, UB2.OUT_DIM](up_eps2)
    var up_a2_t = tile[BATCH, UB2.IN_DIM](up_a2)
    var up_z1_t = tile[BATCH, UB1.IN_DIM](up_z1)
    var up_z2_t = tile[BATCH, UB2.IN_DIM](up_z2)
    var dn_mu0_t = tile[BATCH, DB0.OUT_DIM](dn_mu0)
    var dn_eps0_t = tile[BATCH, DB0.OUT_DIM](dn_eps0)
    var dn_a0_t = tile[BATCH, DB0.IN_DIM](dn_a0)
    var dn_mu1_t = tile[BATCH, DB1.OUT_DIM](dn_mu1)
    var dn_eps1_t = tile[BATCH, DB1.OUT_DIM](dn_eps1)
    var dn_a1_t = tile[BATCH, DB1.IN_DIM](dn_a1)
    var dn_mu2_t = tile[BATCH, DB2.OUT_DIM](dn_mu2)
    var dn_eps2_t = tile[BATCH, DB2.OUT_DIM](dn_eps2)
    var dn_a2_t = tile[BATCH, DB2.IN_DIM](dn_a2)
    var dn_z1_t = tile[BATCH, DB1.IN_DIM](dn_z1)
    var dn_z2_t = tile[BATCH, DB2.IN_DIM](dn_z2)
    # generation chain re-views (consuming block typed)
    var dn_mu0_db1 = tile[BATCH, DB1.IN_DIM](dn_mu0)
    var dn_mu1_db2 = tile[BATCH, DB2.IN_DIM](dn_mu1)

    # ── Train ───────────────────────────────────────────────────────────────
    print("\n  epoch | up_loss | dn_loss | test_acc | wall_t (s)")
    print("  ------+---------+---------+----------+-----------")
    var step_num: Int = 0
    var t0 = perf_counter_ns()

    for epoch in range(EPOCHS):
        var ep_up: Float64 = 0.0
        var ep_dn: Float64 = 0.0
        for batch_idx in range(N_TRAIN_BATCHES):
            for i in range(BATCH):
                var sidx = batch_idx * BATCH + i
                for j in range(IMG):
                    image_buf[i * IMG + j] = ds.train_images[sidx * IMG + j]
                for c in range(10):
                    label_buf[i * 10 + c] = Scalar[dtype](0)
                label_buf[i * 10 + Int(ds.train_labels[sidx])] = Scalar[dtype](
                    1.0
                )

            # init shared latents via UP forward
            UB0.predict[BATCH, dtype](image_ub0, up_p0, up_mu0_t, up_a0_t)
            for i in range(BATCH * X0_DIM):
                x0_buf[i] = up_mu0[i]
            UB1.predict[BATCH, dtype](x0_ub1, up_p1, up_mu1_t, up_a1_t)
            for i in range(BATCH * X1_DIM):
                x1_buf[i] = up_mu1[i]

            for _ in range(T_INFER):
                # UP predictions + errors
                UB0.predict[BATCH, dtype](image_ub0, up_p0, up_mu0_t, up_a0_t)
                UB1.predict[BATCH, dtype](x0_ub1, up_p1, up_mu1_t, up_a1_t)
                UB2.predict[BATCH, dtype](x1_ub2, up_p2, up_mu2_t, up_a2_t)
                UB0.eps_compute[BATCH, dtype](x0_ub0, up_mu0_t, up_eps0_t)
                UB1.eps_compute[BATCH, dtype](x1_ub1, up_mu1_t, up_eps1_t)
                UB2.eps_compute[BATCH, dtype](label_ub2, up_mu2_t, up_eps2_t)

                # DOWN predictions + errors
                DB0.predict[BATCH, dtype](label_db0, dn_p0v, dn_mu0_t, dn_a0_t)
                DB1.predict[BATCH, dtype](x1_db1, dn_p1v, dn_mu1_t, dn_a1_t)
                DB2.predict[BATCH, dtype](x0_db2, dn_p2v, dn_mu2_t, dn_a2_t)
                DB0.eps_compute[BATCH, dtype](x1_db0, dn_mu0_t, dn_eps0_t)
                DB1.eps_compute[BATCH, dtype](x0_db1, dn_mu1_t, dn_eps1_t)
                DB2.eps_compute[BATCH, dtype](image_db2, dn_mu2_t, dn_eps2_t)

                # Pull-backs for x0: UP UB1 (ε_up1) + DOWN DB2 (ε_dn2)
                UB1.pull_back[BATCH, dtype](up_eps1_t, up_p1, up_z1_t)
                UB1.act_derivative_mul[BATCH, dtype](x0_ub1, up_z1_t, up_z1_t)
                DB2.pull_back[BATCH, dtype](dn_eps2_t, dn_p2v, dn_z2_t)
                DB2.act_derivative_mul[BATCH, dtype](x0_db2, dn_z2_t, dn_z2_t)

                # Pull-backs for x1: UP UB2 (ε_up2) + DOWN DB1 (ε_dn1)
                UB2.pull_back[BATCH, dtype](up_eps2_t, up_p2, up_z2_t)
                UB2.act_derivative_mul[BATCH, dtype](x1_ub2, up_z2_t, up_z2_t)
                DB1.pull_back[BATCH, dtype](dn_eps1_t, dn_p1v, dn_z1_t)
                DB1.act_derivative_mul[BATCH, dtype](x1_db1, dn_z1_t, dn_z1_t)

                # dx0 = α_up·(ε_up0 − z_up1) + α_down·(ε_dn1 − z_dn2)
                for i in range(BATCH * X0_DIM):
                    var u = Float64(up_eps0[i]) - Float64(up_z1[i])
                    var d = Float64(dn_eps1[i]) - Float64(dn_z2[i])
                    dx0[i] = Scalar[dtype](ALPHA_UP * u + ALPHA_DOWN * d)
                # dx1 = α_up·(ε_up1 − z_up2) + α_down·(ε_dn0 − z_dn1)
                for i in range(BATCH * X1_DIM):
                    var u = Float64(up_eps1[i]) - Float64(up_z2[i])
                    var d = Float64(dn_eps0[i]) - Float64(dn_z1[i])
                    dx1[i] = Scalar[dtype](ALPHA_UP * u + ALPHA_DOWN * d)

                for i in range(BATCH * X0_DIM):
                    x0_buf[i] = x0_buf[i] - Scalar[dtype](LR_X) * dx0[i]
                for i in range(BATCH * X1_DIM):
                    x1_buf[i] = x1_buf[i] - Scalar[dtype](LR_X) * dx1[i]

            # weight grads (post-inference ε)
            UB0.weight_grad[BATCH, dtype](up_eps0_t, up_a0_t, up_g0)
            UB1.weight_grad[BATCH, dtype](up_eps1_t, up_a1_t, up_g1)
            UB2.weight_grad[BATCH, dtype](up_eps2_t, up_a2_t, up_g2)
            DB0.weight_grad[BATCH, dtype](dn_eps0_t, dn_a0_t, dn_g0)
            DB1.weight_grad[BATCH, dtype](dn_eps1_t, dn_a1_t, dn_g1)
            DB2.weight_grad[BATCH, dtype](dn_eps2_t, dn_a2_t, dn_g2)

            for i in range(DOWN_PARAM_SIZE):
                dn_grads_buf[i] = dn_grads_buf[i] * Scalar[dtype](ALPHA_DOWN)

            step_num += 1
            OPT.step[UP_PARAM_SIZE, dtype](
                up_params, up_grads, up_os, up_og, step_num
            )
            OPT.step[DOWN_PARAM_SIZE, dtype](
                dn_params, dn_grads, dn_os, dn_og, step_num
            )

            var sl: Float64 = 0
            for i in range(BATCH * 10):
                var v = Float64(up_eps2[i])
                sl += v * v
            ep_up += sl * 0.5
            var rl: Float64 = 0
            for i in range(BATCH * IMG):
                var v = Float64(dn_eps2[i])
                rl += v * v
            ep_dn += rl * 0.5

        # accuracy (UP forward_eval — ConvPCBlock + PCBlock all conform)
        var correct: Int = 0
        var pred_buf = alloc[Scalar[dtype]](BATCH * 10).as_unsafe_any_origin()
        memset(pred_buf, 0, BATCH * 10)
        var pred_t = tile[BATCH, UP_NET.OUT_DIM](pred_buf)
        for tb in range(N_TEST_BATCHES):
            for i in range(BATCH):
                var sidx = tb * BATCH + i
                for j in range(IMG):
                    image_buf[i * IMG + j] = ds.test_images[sidx * IMG + j]
            UP_NET.forward_eval[BATCH, dtype](image_ub0, up_params, pred_t)
            for i in range(BATCH):
                var best_c: Int = 0
                var best_v = Float64(pred_buf[i * 10])
                for c in range(1, 10):
                    var v = Float64(pred_buf[i * 10 + c])
                    if v > best_v:
                        best_v = v
                        best_c = c
                if best_c == Int(ds.test_labels[tb * BATCH + i]):
                    correct += 1
        var acc = Float64(correct) / Float64(N_TEST_BATCHES * BATCH)
        pred_buf.free()

        var el = Float64(perf_counter_ns() - t0) / 1e9
        print(
            "    ",
            epoch,
            "  ",
            ep_up / Float64(N_TRAIN_BATCHES),
            "  ",
            ep_dn / Float64(N_TRAIN_BATCHES),
            "  ",
            acc,
            "  ",
            el,
        )

    var total_t = Float64(perf_counter_ns() - t0) / 1e9
    print("\n  total train time:", total_t, "s")

    # ── Generation: one-hot label → DOWN forward (hand-wired) ───────────────
    for i in range(BATCH * 10):
        label_buf[i] = Scalar[dtype](0)
    for c in range(10):
        label_buf[c * 10 + c] = Scalar[dtype](1.0)
    DB0.predict[BATCH, dtype](label_db0, dn_p0v, dn_mu0_t, dn_a0_t)  # → x1
    DB1.predict[BATCH, dtype](dn_mu0_db1, dn_p1v, dn_mu1_t, dn_a1_t)  # → x0
    DB2.predict[BATCH, dtype](dn_mu1_db2, dn_p2v, dn_mu2_t, dn_a2_t)  # → image

    var class_diff: Float64 = 0
    for j in range(IMG):
        var d = Float64(dn_mu2[0 * IMG + j]) - Float64(dn_mu2[9 * IMG + j])
        class_diff += d * d
    class_diff /= Float64(IMG)
    print("  per-pixel MSE class-0 vs class-9:", class_diff)

    # ── Visualize: the decoder regresses pixels in [0,1] via MSE, so show μ
    #    directly (NOT sigmoid, which compresses [0,0.8] to a washed-out gray).
    #    Two variants: clamp[0,1] (clean background) and per-image min/max
    #    stretch (max stroke contrast, but amplifies background speckle). ─────
    var gen_sig = alloc[Scalar[dtype]](10 * IMG).as_unsafe_any_origin()  # clamp[0,1] — primary
    var gen_stretch = alloc[Scalar[dtype]](10 * IMG).as_unsafe_any_origin()  # min/max stretch
    for c in range(10):
        var lo = Float64(dn_mu2[c * IMG])
        var hi = lo
        for j in range(IMG):
            var v = Float64(dn_mu2[c * IMG + j])
            if v < lo:
                lo = v
            if v > hi:
                hi = v
        var rng = hi - lo
        if rng < 1e-6:
            rng = 1.0
        for j in range(IMG):
            var v = Float64(dn_mu2[c * IMG + j])
            var vc = v
            if vc < 0.0:
                vc = 0.0
            if vc > 1.0:
                vc = 1.0
            gen_sig[c * IMG + j] = Scalar[dtype](vc)
            gen_stretch[c * IMG + j] = Scalar[dtype]((v - lo) / rng)

    var real_digits = alloc[Scalar[dtype]](10 * IMG).as_unsafe_any_origin()
    var found = alloc[UInt8](10).as_unsafe_any_origin()
    memset(found, 0, 10)
    var fc: Int = 0
    for i in range(MNIST.N_TEST):
        var c = Int(ds.test_labels[i])
        if found[c] == 0:
            for j in range(IMG):
                real_digits[c * IMG + j] = ds.test_images[i * IMG + j]
            found[c] = 1
            fc += 1
            if fc == 10:
                break
    found.free()

    var labels = List[String]()
    for i in range(10):
        labels.append(String(i))
    save_image_row(
        "pcn_conv_generated_digits.ppm",
        gen_sig,
        n=10,
        height=28,
        width=28,
        channels=1,
        vmin=0.0,
        vmax=1.0,
        pixel_scale=4,
        labels=labels,
    )
    save_image_row(
        "pcn_conv_generated_digits_stretched.ppm",
        gen_stretch,
        n=10,
        height=28,
        width=28,
        channels=1,
        vmin=0.0,
        vmax=1.0,
        pixel_scale=4,
        labels=labels,
    )
    save_image_row(
        "pcn_conv_real_digits.ppm",
        real_digits,
        n=10,
        height=28,
        width=28,
        channels=1,
        vmin=0.0,
        vmax=1.0,
        pixel_scale=4,
        labels=labels,
    )
    save_reconstruction_grid(
        "pcn_conv_real_vs_generated.ppm",
        real_digits,
        gen_sig,
        n=10,
        height=28,
        width=28,
        channels=1,
        vmin=0.0,
        vmax=1.0,
    )
    print(
        "  saved: pcn_conv_generated_digits{,_stretched}.ppm "
        " pcn_conv_real_vs_generated.ppm"
    )
    gen_sig.free()
    gen_stretch.free()
    real_digits.free()

    # ── Verdict ─────────────────────────────────────────────────────────────
    var final_correct: Int = 0
    var fp_buf = alloc[Scalar[dtype]](BATCH * 10).as_unsafe_any_origin()
    memset(fp_buf, 0, BATCH * 10)
    var fp_t = tile[BATCH, UP_NET.OUT_DIM](fp_buf)
    for tb in range(N_TEST_BATCHES):
        for i in range(BATCH):
            var sidx = tb * BATCH + i
            for j in range(IMG):
                image_buf[i * IMG + j] = ds.test_images[sidx * IMG + j]
        UP_NET.forward_eval[BATCH, dtype](image_ub0, up_params, fp_t)
        for i in range(BATCH):
            var best_c: Int = 0
            var best_v = Float64(fp_buf[i * 10])
            for c in range(1, 10):
                var v = Float64(fp_buf[i * 10 + c])
                if v > best_v:
                    best_v = v
                    best_c = c
            if best_c == Int(ds.test_labels[tb * BATCH + i]):
                final_correct += 1
    var final_acc = Float64(final_correct) / Float64(N_TEST_BATCHES * BATCH)
    fp_buf.free()
    print("\n  Final UP test accuracy:", final_acc)
    print("  Generation diversity (class-0 vs 9 MSE):", class_diff)
    if final_acc >= 0.60:
        print("\n  [PASS] bidirectional conv-PC")
    else:
        print("\n  [FAIL] accuracy too low")
        raise Error("bidirectional conv-PC failed")
