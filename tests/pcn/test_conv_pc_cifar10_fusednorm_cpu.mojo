"""Conv-PCN CIFAR-10 with FUSED input-side RMSNorm (CPU) — the conv+norm test.

Normalization folded into the conv blocks (NormConvPCBlock) — so the net keeps
the SAME 5 levels as the unnormalized baseline (no extra latent), unlike the
separate-norm-level experiments (8 levels, which slowed inference and capped
accuracy). This isolates the question: with inference depth held fixed, does
normalizing inter-layer activations help?

  ConvPCBlock[3, 32, 3, 2, 1, 32, 32, PCIdentity]       # raw image conv → 32×16×16
  NormConvPCBlock[32, 64, 3, 2, 1, 16, 16, PCReLU]      # norm 32ch input → conv → 64×8×8
  NormConvPCBlock[64, 64, 3, 2, 1, 8,  8,  PCReLU]      # norm 64ch input → conv → 64×4×4
  PCBlock[1024, 256, PCReLU]
  PCBlock[256, 10, PCIdentity]

Baselines: unnormalized 46.5% @ LR 4e-4 (DIVERGES @ 1e-3); separate norm levels
~0.36. This runs at 1e-3 — the rate that diverges without any norm.

Run:
    pixi run mojo run -I . tests/pcn/test_conv_pc_cifar10_fusednorm_cpu.mojo
"""

from std.memory import alloc, memset
from std.time import perf_counter_ns
from noeira.core.tiles import tile

from noeira.nn.constants import DT as dtype
from noeira.experimental.pcn.pc_initializer import PCXavier
from noeira.experimental.pcn.pc_optimizer import PCAdam
from noeira.nn.datasets.cifar10 import CIFAR10
from noeira.experimental.pcn import (
    PCBlock,
    PCSequential,
    PCIdentity,
    PCReLU,
    PCTrainer,
)
from noeira.experimental.pcn.pc_conv_block import ConvPCBlock
from noeira.experimental.pcn.pc_norm_conv_block import NormConvPCBlock
from noeira.core.fmt import fit

comptime BATCH = 125
comptime EPOCHS = 6
comptime T_INFER = 12
comptime LR_X: Float32 = 0.05
comptime ADAM_LR: Float64 = 0.0004  # matched to the unnormalized baseline (0.465)
comptime N_TRAIN_SAMPLES = 20000
comptime N_TEST_SAMPLES = 1000
comptime N_TRAIN_BATCHES = N_TRAIN_SAMPLES // BATCH
comptime N_TEST_BATCHES = N_TEST_SAMPLES // BATCH
comptime IN = 3 * 32 * 32
comptime PASS_ACC = 0.13  # NEGATIVE-RESULT experiment, kept for reproducibility.
#   Fused input-side RMSNorm trains (block is FD-validated) but HURTS badly:
#   ~0.17 vs unnormalized 0.465 — worse even than separate norm levels (~0.36).
#   Bar set just above chance (0.10) to assert it runs end-to-end; the finding
#   is that normalization does not help PC conv. See docs/PCN_CONV_DESIGN.md.

comptime NET = PCSequential[
    ConvPCBlock[3, 32, 3, 2, 1, 32, 32, PCIdentity],
    NormConvPCBlock[32, 64, 3, 2, 1, 16, 16, PCReLU],
    NormConvPCBlock[64, 64, 3, 2, 1, 8, 8, PCReLU],
    PCBlock[1024, 256, PCReLU],
    PCBlock[256, 10, PCIdentity],
]
comptime TRAINER = PCTrainer[
    ConvPCBlock[3, 32, 3, 2, 1, 32, 32, PCIdentity],
    NormConvPCBlock[32, 64, 3, 2, 1, 16, 16, PCReLU],
    NormConvPCBlock[64, 64, 3, 2, 1, 8, 8, PCReLU],
    PCBlock[1024, 256, PCReLU],
    PCBlock[256, 10, PCIdentity],
    dtype=dtype,
]
comptime OPT = PCAdam[LR=ADAM_LR]


def main() raises:
    print("Conv-PCN CIFAR-10 + FUSED input-side RMSNorm (CPU)\n")
    print("  loading CIFAR-10 ...")
    var ds = CIFAR10()
    print("  LATENT_DIM=", NET.LATENT_DIM, " (5 levels, same as unnorm)",
          " PARAM_SIZE=", NET.PARAM_SIZE, " ADAM_LR=", ADAM_LR)

    var params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var grads_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    memset(params_buf, 0, NET.PARAM_SIZE)
    memset(grads_buf, 0, NET.PARAM_SIZE)
    var params = tile[NET.PARAM_SIZE](params_buf)
    var grads = tile[NET.PARAM_SIZE](grads_buf)
    NET.pc_init_params[PCXavier, dtype](params)

    var opt_state_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE * OPT.STATE_PER_PARAM).as_unsafe_any_origin()
    var opt_global_buf = alloc[Scalar[dtype]](OPT.GLOBAL_STATE_SIZE).as_unsafe_any_origin()
    memset(opt_state_buf, 0, NET.PARAM_SIZE * OPT.STATE_PER_PARAM)
    memset(opt_global_buf, 0, OPT.GLOBAL_STATE_SIZE)
    var opt_state = tile[NET.PARAM_SIZE, OPT.STATE_PER_PARAM](opt_state_buf)
    var opt_global = tile[OPT.GLOBAL_STATE_SIZE](opt_global_buf)

    var lat_buf = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var mu_eps_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_OUT_DIM).as_unsafe_any_origin()
    var a_below_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var z_below_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var dx_buf_raw = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var eval_out_buf = alloc[Scalar[dtype]](BATCH * NET.OUT_DIM).as_unsafe_any_origin()
    var x_buf = alloc[Scalar[dtype]](BATCH * IN).as_unsafe_any_origin()
    var y_buf = alloc[Scalar[dtype]](BATCH * NET.OUT_DIM).as_unsafe_any_origin()
    memset(lat_buf, 0, BATCH * NET.LATENT_DIM)
    memset(mu_eps_buf_raw, 0, BATCH * NET.SCRATCH_OUT_DIM)
    memset(a_below_buf_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
    memset(z_below_buf_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
    memset(dx_buf_raw, 0, BATCH * NET.LATENT_DIM)

    var latents = tile[BATCH, NET.LATENT_DIM](lat_buf)
    var mu_eps_buf = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_eps_buf_raw)
    var a_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](a_below_buf_raw)
    var z_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](z_below_buf_raw)
    var dx_buf = tile[BATCH, NET.LATENT_DIM](dx_buf_raw)
    var eval_out = tile[BATCH, NET.OUT_DIM](eval_out_buf)
    var x_batch = tile[BATCH, IN](x_buf)
    var y_batch = tile[BATCH, NET.OUT_DIM](y_buf)

    var step_num: Int = 0
    var t0 = perf_counter_ns()
    print("\n  epoch | sup_loss  | train_acc | wall_t (s)")
    print("  ------+-----------+-----------+-----------")
    for epoch in range(EPOCHS):
        var ep_loss: Float64 = 0.0
        for batch_idx in range(N_TRAIN_BATCHES):
            for i in range(BATCH):
                var sidx = batch_idx * BATCH + i
                for j in range(IN):
                    x_buf[i * IN + j] = ds.train_images[sidx * IN + j]
                for c in range(NET.OUT_DIM):
                    y_buf[i * NET.OUT_DIM + c] = 0
                y_buf[i * NET.OUT_DIM + Int(ds.train_labels[sidx])] = 1

            var r = TRAINER.compute_grads_only[BATCH](
                params, grads, latents, mu_eps_buf, a_below_buf,
                z_below_buf, dx_buf, x_batch, y_batch,
                T_infer=T_INFER, lr_x=Scalar[dtype](LR_X),
            )
            ep_loss += r.output_loss_final
            step_num += 1
            OPT.step[NET.PARAM_SIZE, dtype](params, grads, opt_state, opt_global, step_num)

        var tr_correct: Int = 0
        for tb in range(N_TEST_BATCHES):
            for i in range(BATCH):
                var sidx = tb * BATCH + i
                for j in range(IN):
                    x_buf[i * IN + j] = ds.train_images[sidx * IN + j]
            NET.forward_eval[BATCH, dtype](x_batch, params, eval_out)
            for i in range(BATCH):
                var best_c: Int = 0
                var best_v = Float64(eval_out_buf[i * NET.OUT_DIM])
                for c in range(1, NET.OUT_DIM):
                    var v = Float64(eval_out_buf[i * NET.OUT_DIM + c])
                    if v > best_v:
                        best_v = v; best_c = c
                if best_c == Int(ds.train_labels[tb * BATCH + i]):
                    tr_correct += 1
        var tr_acc = Float64(tr_correct) / Float64(N_TEST_BATCHES * BATCH)
        var el = Float64(perf_counter_ns() - t0) / 1e9
        print("    ", epoch, "  ", ep_loss / Float64(N_TRAIN_BATCHES), "  ",
              tr_acc, "  ", el)

    var train_t = Float64(perf_counter_ns() - t0) / 1e9

    var correct: Int = 0
    for tb in range(N_TEST_BATCHES):
        for i in range(BATCH):
            var sidx = tb * BATCH + i
            for j in range(IN):
                x_buf[i * IN + j] = ds.test_images[sidx * IN + j]
        NET.forward_eval[BATCH, dtype](x_batch, params, eval_out)
        for i in range(BATCH):
            var best_c: Int = 0
            var best_v = Float64(eval_out_buf[i * NET.OUT_DIM])
            for c in range(1, NET.OUT_DIM):
                var v = Float64(eval_out_buf[i * NET.OUT_DIM + c])
                if v > best_v:
                    best_v = v; best_c = c
            if best_c == Int(ds.test_labels[tb * BATCH + i]):
                correct += 1
    var acc = Float64(correct) / Float64(N_TEST_BATCHES * BATCH)

    print("\n  test accuracy =", acc, " (", correct, "/",
          N_TEST_BATCHES * BATCH, ")")
    print("  train time =", fit(String(train_t), 7), "s")
    print("  (unnorm 46.5% @4e-4 / diverges @1e-3 ; separate-norm-level ~0.36)")
    print("")
    if acc >= PASS_ACC:
        print("✅ PASS (runs) — NEGATIVE RESULT: fused input-norm HURTS (acc",
              acc, "vs unnormalized 0.465); normalization doesn't help PC conv")
    else:
        print("❌ FAIL — acc", acc, "<", PASS_ACC, "(below chance — likely a bug)")
        raise Error("fused-norm conv-PCN below chance")
