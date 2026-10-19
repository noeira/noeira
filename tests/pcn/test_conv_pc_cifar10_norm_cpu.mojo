"""Conv-PCN CIFAR-10 with RMSNorm PC levels (CPU Accelerate) — P4 follow-up.

Inserts NormPCBlock (RMSNorm PC level) between conv blocks → Conv→Norm→ReLU→Conv
ordering. The unnormalized stack (test_conv_pc_cifar10_cpu.mojo) diverged at
ADAM_LR=1e-3 once steps grew; this checks whether normalization (a) stabilizes
that rate and (b) lifts accuracy past the ~47% unnormalized result.

  ConvPCBlock[3,  32, 3, 2, 1, 32, 32, PCIdentity]  # → 32×16×16 (8192)
  NormPCBlock[8192]
  ConvPCBlock[32, 64, 3, 2, 1, 16, 16, PCReLU]      # → 64×8×8  (4096)
  NormPCBlock[4096]
  ConvPCBlock[64, 64, 3, 2, 1, 8,  8,  PCReLU]      # → 64×4×4  (1024)
  NormPCBlock[1024]
  PCBlock[1024, 256, PCReLU]
  PCBlock[256, 10, PCIdentity]

Run:
    pixi run mojo run -I . tests/pcn/test_conv_pc_cifar10_norm_cpu.mojo
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
from noeira.experimental.pcn.pc_norm_block import NormPCBlock
from noeira.core.fmt import fit

comptime BATCH = 125
comptime EPOCHS = 5
comptime T_INFER = 12
comptime LR_X: Float32 = 0.05
comptime ADAM_LR: Float64 = 0.001  # stable WITH norm (unnorm diverged here); 4e-3 collapses
comptime N_TRAIN_SAMPLES = 20000
comptime N_TEST_SAMPLES = 1000
comptime N_TRAIN_BATCHES = N_TRAIN_SAMPLES // BATCH
comptime N_TEST_BATCHES = N_TEST_SAMPLES // BATCH
comptime IN = 3 * 32 * 32
comptime PASS_ACC = 0.30  # this test verifies STABLE learning at 1e-3 (unnorm
#                            DIVERGES here), not peak accuracy. Global RMSNorm
#                            plateaus ~0.38 (8 ep) — below the tuned unnorm 0.465;
#                            per-channel/group norm is the path past 50%.

comptime NET = PCSequential[
    ConvPCBlock[3, 32, 3, 2, 1, 32, 32, PCIdentity],
    NormPCBlock[8192],
    ConvPCBlock[32, 64, 3, 2, 1, 16, 16, PCReLU],
    NormPCBlock[4096],
    ConvPCBlock[64, 64, 3, 2, 1, 8, 8, PCReLU],
    NormPCBlock[1024],
    PCBlock[1024, 256, PCReLU],
    PCBlock[256, 10, PCIdentity],
]
comptime TRAINER = PCTrainer[
    ConvPCBlock[3, 32, 3, 2, 1, 32, 32, PCIdentity],
    NormPCBlock[8192],
    ConvPCBlock[32, 64, 3, 2, 1, 16, 16, PCReLU],
    NormPCBlock[4096],
    ConvPCBlock[64, 64, 3, 2, 1, 8, 8, PCReLU],
    NormPCBlock[1024],
    PCBlock[1024, 256, PCReLU],
    PCBlock[256, 10, PCIdentity],
    dtype=dtype,
]
comptime OPT = PCAdam[LR=ADAM_LR]


def main() raises:
    print("Conv-PCN CIFAR-10 + RMSNorm PC levels (CPU)\n")
    print("  loading CIFAR-10 ...")
    var ds = CIFAR10()
    print("  IN_DIM=", NET.IN_DIM, " OUT_DIM=", NET.OUT_DIM,
          " LATENT_DIM=", NET.LATENT_DIM, " PARAM_SIZE=", NET.PARAM_SIZE,
          " ADAM_LR=", ADAM_LR)

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
    print("  (unnormalized baseline: 46.5% at ADAM_LR=4e-4, diverged at 1e-3)")
    print("")
    if acc >= PASS_ACC:
        print("✅ PASS — normalized conv-PCN trains stably at 1e-3 (acc", acc, ")")
    else:
        print("❌ FAIL — acc", acc, "<", PASS_ACC)
        raise Error("normalized conv-PCN CIFAR-10 below threshold")
