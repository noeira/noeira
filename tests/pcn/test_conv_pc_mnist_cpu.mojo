"""Conv-PCN MNIST on CPU (Accelerate fast path) — see docs/PCN_CONV_DESIGN.md.

Same net/budget as test_conv_pc_mnist_gpu.mojo but runs the CPU PCTrainer path
(compute_grads_only → Adam.step) using the im2col + Apple Accelerate sgemm
ConvPCBlock CPU kernels. On Apple, this is expected to BEAT the GPU run — the
conv ops are small and GPU kernel-launch overhead dominates.

Run:
    pixi run mojo run -I . tests/pcn/test_conv_pc_mnist_cpu.mojo
"""

from std.memory import alloc, memset
from std.time import perf_counter_ns
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
    PCTrainer,
)
from noeira.experimental.pcn.pc_conv_block import ConvPCBlock
from noeira.core.fmt import fit

comptime BATCH = 64
comptime EPOCHS = 3
comptime T_INFER = 20
comptime LR_X: Float32 = 0.05
comptime ADAM_LR: Float64 = 0.001
comptime N_TRAIN_SAMPLES = 5000
comptime N_TEST_SAMPLES = 1000
comptime N_TRAIN_BATCHES = N_TRAIN_SAMPLES // BATCH
comptime N_TEST_BATCHES = N_TEST_SAMPLES // BATCH
comptime PASS_ACC = 0.80

comptime NET = PCSequential[
    ConvPCBlock[1, 8, 3, 2, 1, 28, 28, PCIdentity],
    ConvPCBlock[8, 16, 3, 2, 1, 14, 14, PCReLU],
    PCBlock[784, 128, PCReLU],
    PCBlock[128, 10, PCIdentity],
]
comptime TRAINER = PCTrainer[
    ConvPCBlock[1, 8, 3, 2, 1, 28, 28, PCIdentity],
    ConvPCBlock[8, 16, 3, 2, 1, 14, 14, PCReLU],
    PCBlock[784, 128, PCReLU],
    PCBlock[128, 10, PCIdentity],
    dtype=dtype,
]
comptime OPT = PCAdam[LR=ADAM_LR]


def main() raises:
    print("Conv-PCN MNIST on CPU (Accelerate fast path)\n")
    print("  loading MNIST ...")
    var ds = MNIST()
    print("  PARAM_SIZE=", NET.PARAM_SIZE, " LATENT_DIM=", NET.LATENT_DIM)

    var params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var grads_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    memset(params_buf, 0, NET.PARAM_SIZE)
    memset(grads_buf, 0, NET.PARAM_SIZE)
    var params = tile[NET.PARAM_SIZE](params_buf)
    var grads = tile[NET.PARAM_SIZE](grads_buf)
    NET.pc_init_params[PCXavier, dtype](params)

    var opt_state_buf = alloc[Scalar[dtype]](
        NET.PARAM_SIZE * OPT.STATE_PER_PARAM
    ).as_unsafe_any_origin()
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
    var x_buf = alloc[Scalar[dtype]](BATCH * NET.IN_DIM).as_unsafe_any_origin()
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
    var x_batch = tile[BATCH, NET.IN_DIM](x_buf)
    var y_batch = tile[BATCH, NET.OUT_DIM](y_buf)

    var step_num: Int = 0
    var t0 = perf_counter_ns()
    for epoch in range(EPOCHS):
        for batch_idx in range(N_TRAIN_BATCHES):
            for i in range(BATCH):
                var sidx = batch_idx * BATCH + i
                for j in range(NET.IN_DIM):
                    x_buf[i * NET.IN_DIM + j] = ds.train_images[
                        sidx * NET.IN_DIM + j
                    ]
                for c in range(NET.OUT_DIM):
                    y_buf[i * NET.OUT_DIM + c] = 0
                y_buf[i * NET.OUT_DIM + Int(ds.train_labels[sidx])] = 1

            var r = TRAINER.compute_grads_only[BATCH](
                params, grads, latents, mu_eps_buf, a_below_buf,
                z_below_buf, dx_buf, x_batch, y_batch,
                T_infer=T_INFER, lr_x=Scalar[dtype](LR_X),
            )
            _ = r
            step_num += 1
            OPT.step[NET.PARAM_SIZE, dtype](
                params, grads, opt_state, opt_global, step_num
            )

    var train_t = Float64(perf_counter_ns() - t0) / 1e9

    var correct: Int = 0
    for tb in range(N_TEST_BATCHES):
        for i in range(BATCH):
            var sidx = tb * BATCH + i
            for j in range(NET.IN_DIM):
                x_buf[i * NET.IN_DIM + j] = ds.test_images[sidx * NET.IN_DIM + j]
        NET.forward_eval[BATCH, dtype](x_batch, params, eval_out)
        for i in range(BATCH):
            var sidx = tb * BATCH + i
            var best_c: Int = 0
            var best_v = Float64(eval_out_buf[i * NET.OUT_DIM])
            for c in range(1, NET.OUT_DIM):
                var v = Float64(eval_out_buf[i * NET.OUT_DIM + c])
                if v > best_v:
                    best_v = v
                    best_c = c
            if best_c == Int(ds.test_labels[sidx]):
                correct += 1
    var acc = Float64(correct) / Float64(N_TEST_BATCHES * BATCH)

    print("\n  test accuracy =", acc, " (", correct, "/",
          N_TEST_BATCHES * BATCH, ")")
    print("  train time =", fit(String(train_t), 7), "s  (GPU ref ~17.6s)")
    print("")
    if acc >= PASS_ACC:
        print("✅ PASS — CPU conv-PCN learns MNIST (acc", acc, ")")
    else:
        print("❌ FAIL — acc", acc, "<", PASS_ACC)
        raise Error("CPU conv-PCN MNIST below threshold")
