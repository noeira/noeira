"""Conv-PCN MNIST lighthouse (P3, GPU) — see docs/PCN_CONV_DESIGN.md.

Trains an all-convolutional PC network on MNIST through the real PCTrainer GPU
path (compute_grads_only_gpu → Adam.step_gpu), exercising the P2 ConvPCBlock
GPU kernels end-to-end. The research question: does the Bogacz local energy
rule scale to convolutions in our stack and learn real images?

Architecture (all-conv → flat readout, no Flatten op — conv OUT_DIM is already
the flat feature vector the next block consumes):
  ConvPCBlock[1,  8, 3, 2, 1, 28, 28, PCIdentity]  # 1×28×28 (784) → 8×14×14 (1568)
  ConvPCBlock[8, 16, 3, 2, 1, 14, 14, PCReLU]      # 8×14×14    → 16×7×7  (784)
  PCBlock[784, 128, PCReLU]                         # 784 → 128
  PCBlock[128, 10, PCIdentity]                      # 128 → 10 logits

Pass: test accuracy ≥ 80% on a subset (CPU-budget-style; the MLP PCN baseline
targets the same). What matters is "PC-conv learns real images, not toys".

Run (Apple):
    pixi run -e apple mojo run -I . tests/pcn/test_conv_pc_mnist_gpu.mojo
"""

from std.memory import alloc, memset
from std.time import perf_counter_ns
from noeira.core.tiles import tile
from max.gpu.host import DeviceContext

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
    print("Conv-PCN MNIST lighthouse (P3, GPU)\n")
    print("  loading MNIST ...")
    var ds = MNIST()
    print("  train=", ds.num_train, " test=", ds.num_test,
          " IN_DIM=", NET.IN_DIM, " OUT_DIM=", NET.OUT_DIM,
          " LATENT_DIM=", NET.LATENT_DIM, " PARAM_SIZE=", NET.PARAM_SIZE)

    var ctx = DeviceContext()

    # ── Params (host init → device) ──────────────────────────────────────────
    var params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    memset(params_buf, 0, NET.PARAM_SIZE)
    var params_host_lt = tile[NET.PARAM_SIZE](params_buf)
    NET.pc_init_params[PCXavier, dtype](params_host_lt)

    var params_d = ctx.enqueue_create_buffer[dtype](NET.PARAM_SIZE)
    var grads_d = ctx.enqueue_create_buffer[dtype](NET.PARAM_SIZE)
    var p_host = ctx.enqueue_create_host_buffer[dtype](NET.PARAM_SIZE)
    for i in range(NET.PARAM_SIZE):
        p_host.unsafe_ptr()[i] = params_buf[i]
    ctx.enqueue_copy(params_d, p_host)

    # ── Adam state (zero on device) ──────────────────────────────────────────
    comptime OPT_STATE = NET.PARAM_SIZE * OPT.STATE_PER_PARAM
    var opt_state_d = ctx.enqueue_create_buffer[dtype](OPT_STATE)
    var opt_global_d = ctx.enqueue_create_buffer[dtype](OPT.GLOBAL_STATE_SIZE)
    var zero_state = ctx.enqueue_create_host_buffer[dtype](OPT_STATE)
    for i in range(OPT_STATE):
        zero_state.unsafe_ptr()[i] = 0
    ctx.enqueue_copy(opt_state_d, zero_state)
    var zero_global = ctx.enqueue_create_host_buffer[dtype](
        OPT.GLOBAL_STATE_SIZE
    )
    for i in range(OPT.GLOBAL_STATE_SIZE):
        zero_global.unsafe_ptr()[i] = 0
    # Slot 0 = step counter (0); slot 1 = lr_scale (MUST be 1.0 — the GPU Adam
    # kernel computes lr = base_lr * lr_scale, so a zero here freezes training).
    zero_global.unsafe_ptr()[1] = 1
    ctx.enqueue_copy(opt_global_d, zero_global)

    # ── Latents + scratch (device) ───────────────────────────────────────────
    var lat_d = ctx.enqueue_create_buffer[dtype](BATCH * NET.LATENT_DIM)
    var mu_eps_d = ctx.enqueue_create_buffer[dtype](BATCH * NET.SCRATCH_OUT_DIM)
    var a_below_d = ctx.enqueue_create_buffer[dtype](BATCH * NET.SCRATCH_IN_DIM)
    var z_below_d = ctx.enqueue_create_buffer[dtype](BATCH * NET.SCRATCH_IN_DIM)
    var dx_d = ctx.enqueue_create_buffer[dtype](BATCH * NET.LATENT_DIM)
    var x_in_d = ctx.enqueue_create_buffer[dtype](BATCH * NET.IN_DIM)
    var y_target_d = ctx.enqueue_create_buffer[dtype](BATCH * NET.OUT_DIM)
    var eval_out_d = ctx.enqueue_create_buffer[dtype](BATCH * NET.OUT_DIM)

    # ── Reusable host staging buffers ────────────────────────────────────────
    var x_host = ctx.enqueue_create_host_buffer[dtype](BATCH * NET.IN_DIM)
    var y_host = ctx.enqueue_create_host_buffer[dtype](BATCH * NET.OUT_DIM)
    var eval_host = ctx.enqueue_create_host_buffer[dtype](BATCH * NET.OUT_DIM)

    # ── Device LayoutTensor views ────────────────────────────────────────────
    var params_t = tile[NET.PARAM_SIZE](params_d)
    var grads_t = tile[NET.PARAM_SIZE](grads_d)
    var opt_state_t = tile[NET.PARAM_SIZE, OPT.STATE_PER_PARAM](opt_state_d)
    var opt_global_t = tile[OPT.GLOBAL_STATE_SIZE](opt_global_d)
    var lat_t = tile[BATCH, NET.LATENT_DIM](lat_d)
    var mu_eps_t = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_eps_d)
    var a_below_t = tile[BATCH, NET.SCRATCH_IN_DIM](a_below_d)
    var z_below_t = tile[BATCH, NET.SCRATCH_IN_DIM](z_below_d)
    var dx_t = tile[BATCH, NET.LATENT_DIM](dx_d)
    var x_in_t = tile[BATCH, NET.IN_DIM](x_in_d)
    var y_target_t = tile[BATCH, NET.OUT_DIM](y_target_d)
    var eval_out_t = tile[BATCH, NET.OUT_DIM](eval_out_d)

    # ── Train ─────────────────────────────────────────────────────────────────
    var step_num: Int = 0
    var t0 = perf_counter_ns()
    print("\n  epoch | batch | train_t (s)")
    print("  ------+-------+------------")
    for epoch in range(EPOCHS):
        for batch_idx in range(N_TRAIN_BATCHES):
            # Stage minibatch (image + one-hot) on host, upload.
            for i in range(BATCH):
                var sample_idx = batch_idx * BATCH + i
                for j in range(NET.IN_DIM):
                    x_host.unsafe_ptr()[i * NET.IN_DIM + j] = ds.train_images[
                        sample_idx * NET.IN_DIM + j
                    ]
                for c in range(NET.OUT_DIM):
                    y_host.unsafe_ptr()[i * NET.OUT_DIM + c] = 0
                var lbl = Int(ds.train_labels[sample_idx])
                y_host.unsafe_ptr()[i * NET.OUT_DIM + lbl] = 1
            ctx.enqueue_copy(x_in_d, x_host)
            ctx.enqueue_copy(y_target_d, y_host)

            TRAINER.compute_grads_only_gpu[BATCH](
                ctx,
                params_t,
                grads_t,
                lat_t,
                mu_eps_t,
                a_below_t,
                z_below_t,
                dx_t,
                x_in_t,
                y_target_t,
                T_infer=T_INFER,
                lr_x=Scalar[dtype](LR_X),
            )
            step_num += 1
            OPT.step_gpu[NET.PARAM_SIZE, dtype](
                ctx, params_t, grads_t, opt_state_t, opt_global_t, step_num
            )

            if batch_idx == 0 or (batch_idx + 1) % 20 == 0:
                ctx.synchronize()
                var elapsed = Float64(perf_counter_ns() - t0) / 1e9
                print("    ", epoch, "  ", batch_idx, "  ",
                      fit(String(elapsed), 7))
        ctx.synchronize()

    # ── Eval ──────────────────────────────────────────────────────────────────
    var correct: Int = 0
    for tb in range(N_TEST_BATCHES):
        for i in range(BATCH):
            var sample_idx = tb * BATCH + i
            for j in range(NET.IN_DIM):
                x_host.unsafe_ptr()[i * NET.IN_DIM + j] = ds.test_images[
                    sample_idx * NET.IN_DIM + j
                ]
        ctx.enqueue_copy(x_in_d, x_host)
        NET.forward_eval_gpu[BATCH, dtype](
            ctx, x_in_t, params_t, eval_out_t, mu_eps_t, a_below_t
        )
        ctx.enqueue_copy(eval_host, eval_out_d)
        ctx.synchronize()
        for i in range(BATCH):
            var sample_idx = tb * BATCH + i
            var best_c: Int = 0
            var best_v = Float64(eval_host.unsafe_ptr()[i * NET.OUT_DIM])
            for c in range(1, NET.OUT_DIM):
                var v = Float64(eval_host.unsafe_ptr()[i * NET.OUT_DIM + c])
                if v > best_v:
                    best_v = v
                    best_c = c
            if best_c == Int(ds.test_labels[sample_idx]):
                correct += 1

    var acc = Float64(correct) / Float64(N_TEST_BATCHES * BATCH)
    var total_t = Float64(perf_counter_ns() - t0) / 1e9
    print("\n  test accuracy =", acc, " (", correct, "/",
          N_TEST_BATCHES * BATCH, ")")
    print("  total train+eval time =", fit(String(total_t), 7), "s")

    print("")
    if acc >= PASS_ACC:
        print("✅ PASS — conv-PCN learns MNIST (acc", acc, "≥", PASS_ACC, ")")
    else:
        print("❌ FAIL — acc", acc, "<", PASS_ACC)
        raise Error("conv-PCN MNIST below threshold")
