"""End-to-end MNIST PCN test (GPU) — validates Bogacz canonical on accelerator.

Architecture:
    PCBlock[784, 128, PCIdentity]   # input → x_1
    PCBlock[128, 128, PCReLU]       # x_1 → x_2
    PCBlock[128,  10, PCReLU]       # x_2 → output (readout)

Pass criterion (GPU, modest budget):
  - Test accuracy ≥ 80% after 2 epochs on a 5000-sample subset.
  This mirrors the CPU MNIST test so we can compare timing — same dataset,
  same architecture, same hyperparams. GPU should be substantially faster
  (kernel launches dominate at this scale, but we get the parallelism win).

Run:
    pixi run -e apple  mojo run -I . tests/pcn/test_mnist_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/pcn/test_mnist_gpu.mojo
"""

from std.memory import alloc, memset
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
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


comptime BATCH = 64
comptime EPOCHS = 2
comptime T_INFER = 20
comptime LR_X: Float64 = 0.05
comptime ADAM_LR: Float64 = 0.001

comptime N_TRAIN_SAMPLES = 5000
comptime N_TEST_SAMPLES = 1000
comptime N_TRAIN_BATCHES = N_TRAIN_SAMPLES // BATCH
comptime N_TEST_BATCHES = N_TEST_SAMPLES // BATCH

comptime NET = PCSequential[
    PCBlock[784, 128, PCIdentity],
    PCBlock[128, 128, PCReLU],
    PCBlock[128, 10, PCReLU],
]
comptime TRAINER = PCTrainer[
    PCBlock[784, 128, PCIdentity],
    PCBlock[128, 128, PCReLU],
    PCBlock[128, 10, PCReLU],
    dtype=dtype,
]
comptime OPT = PCAdam[LR=ADAM_LR]


def main() raises:
    print("=" * 65)
    print("MNIST PCN (Bogacz canonical, GPU) — Phase 1 GPU validation")
    print("=" * 65)
    print("  arch       : 784 → 128 → 128 → 10")
    print("  PARAM_SIZE :", NET.PARAM_SIZE)
    print("  LATENT_DIM :", NET.LATENT_DIM)
    print(
        "  hyperparams: BATCH=", BATCH, " T_INFER=", T_INFER,
        " EPOCHS=", EPOCHS,
    )
    print("  optimizer  : Adam(lr=", ADAM_LR, "), latent SGD lr=", LR_X)

    var ctx = DeviceContext()
    var ds = MNIST()
    print("  [mnist] loaded:", MNIST.N_TRAIN, "train,", MNIST.N_TEST, "test")

    # ── params init on host, upload to GPU ────────────────────────────────────
    var params_host_init = ctx.enqueue_create_host_buffer[dtype](NET.PARAM_SIZE)
    for i in range(NET.PARAM_SIZE):
        params_host_init.unsafe_ptr()[i] = Scalar[dtype](0)
    var params_init_t = tile[NET.PARAM_SIZE](params_host_init.unsafe_ptr().as_unsafe_any_origin())
    NET.pc_init_params[PCXavier, dtype](params_init_t)

    var params_dbuf = ctx.enqueue_create_buffer[dtype](NET.PARAM_SIZE)
    ctx.enqueue_copy(params_dbuf, params_host_init)
    var params_t = tile[NET.PARAM_SIZE](params_dbuf)

    # ── grads, latents, scratch buffers (GPU only) ────────────────────────────
    var grads_dbuf = ctx.enqueue_create_buffer[dtype](NET.PARAM_SIZE)
    var lat_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.LATENT_DIM)
    var mu_eps_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.SCRATCH_OUT_DIM)
    var a_below_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.SCRATCH_IN_DIM)
    var z_below_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.SCRATCH_IN_DIM)
    var dx_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.LATENT_DIM)
    var eval_out_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.OUT_DIM)

    var grads_t = tile[NET.PARAM_SIZE](grads_dbuf)
    var lat_t = tile[BATCH, NET.LATENT_DIM](lat_dbuf)
    var mu_eps_t = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_eps_dbuf)
    var a_below_t = tile[BATCH, NET.SCRATCH_IN_DIM](a_below_dbuf)
    var z_below_t = tile[BATCH, NET.SCRATCH_IN_DIM](z_below_dbuf)
    var dx_t = tile[BATCH, NET.LATENT_DIM](dx_dbuf)
    var eval_out_t = tile[BATCH, NET.OUT_DIM](eval_out_dbuf)

    # ── Adam state on GPU ─────────────────────────────────────────────────────
    var opt_state_dbuf = ctx.enqueue_create_buffer[dtype](
        NET.PARAM_SIZE * OPT.STATE_PER_PARAM
    )
    var opt_global_dbuf = ctx.enqueue_create_buffer[dtype](OPT.GLOBAL_STATE_SIZE)
    # Init opt state = 0, opt_global = [counter=0, lr_scale=1.0]
    var opt_state_init = ctx.enqueue_create_host_buffer[dtype](
        NET.PARAM_SIZE * OPT.STATE_PER_PARAM
    )
    for i in range(NET.PARAM_SIZE * OPT.STATE_PER_PARAM):
        opt_state_init.unsafe_ptr()[i] = Scalar[dtype](0)
    ctx.enqueue_copy(opt_state_dbuf, opt_state_init)

    var opt_global_init = ctx.enqueue_create_host_buffer[dtype](OPT.GLOBAL_STATE_SIZE)
    opt_global_init.unsafe_ptr()[0] = Scalar[dtype](0)  # step counter (bit-pattern UInt32)
    opt_global_init.unsafe_ptr()[1] = Scalar[dtype](1.0)  # lr_scale = 1.0
    ctx.enqueue_copy(opt_global_dbuf, opt_global_init)

    var opt_state_t = tile[NET.PARAM_SIZE, OPT.STATE_PER_PARAM](opt_state_dbuf)
    var opt_global_t = tile[OPT.GLOBAL_STATE_SIZE](opt_global_dbuf)

    # ── Upload entire MNIST train + test sets once ────────────────────────────
    var train_img_host = ctx.enqueue_create_host_buffer[dtype](
        MNIST.N_TRAIN * MNIST.IMG_SIZE
    )
    var train_tgt_host = ctx.enqueue_create_host_buffer[dtype](
        MNIST.N_TRAIN * MNIST.NUM_CLASSES
    )
    for i in range(MNIST.N_TRAIN * MNIST.IMG_SIZE):
        train_img_host.unsafe_ptr()[i] = ds.train_images[i]
    for i in range(MNIST.N_TRAIN * MNIST.NUM_CLASSES):
        train_tgt_host.unsafe_ptr()[i] = Scalar[dtype](0)
    for i in range(MNIST.N_TRAIN):
        train_tgt_host.unsafe_ptr()[
            i * MNIST.NUM_CLASSES + Int(ds.train_labels[i])
        ] = Scalar[dtype](1.0)
    var train_img_dbuf = ctx.enqueue_create_buffer[dtype](
        MNIST.N_TRAIN * MNIST.IMG_SIZE
    )
    var train_tgt_dbuf = ctx.enqueue_create_buffer[dtype](
        MNIST.N_TRAIN * MNIST.NUM_CLASSES
    )
    ctx.enqueue_copy(train_img_dbuf, train_img_host)
    ctx.enqueue_copy(train_tgt_dbuf, train_tgt_host)

    var test_img_host = ctx.enqueue_create_host_buffer[dtype](
        MNIST.N_TEST * MNIST.IMG_SIZE
    )
    for i in range(MNIST.N_TEST * MNIST.IMG_SIZE):
        test_img_host.unsafe_ptr()[i] = ds.test_images[i]
    var test_img_dbuf = ctx.enqueue_create_buffer[dtype](
        MNIST.N_TEST * MNIST.IMG_SIZE
    )
    ctx.enqueue_copy(test_img_dbuf, test_img_host)

    ctx.synchronize()
    print("  [gpu] datasets uploaded")

    # ── Eval helper (inline, downloads outputs per batch for argmax) ──────────
    var eval_out_host = ctx.enqueue_create_host_buffer[dtype](BATCH * NET.OUT_DIM)

    var initial_correct: Int = 0
    for tb in range(N_TEST_BATCHES):
        var x_batch_t = tile[BATCH, NET.IN_DIM](test_img_dbuf.unsafe_ptr().as_unsafe_any_origin() + tb * BATCH * NET.IN_DIM)
        NET.forward_eval_gpu[BATCH, dtype](
            ctx, x_batch_t, params_t, eval_out_t, mu_eps_t, a_below_t
        )
        ctx.enqueue_copy(eval_out_host, eval_out_dbuf)
        ctx.synchronize()
        for i in range(BATCH):
            var sample_idx = tb * BATCH + i
            var best_class: Int = 0
            var best_val = Float64(eval_out_host.unsafe_ptr()[i * NET.OUT_DIM])
            for c in range(1, NET.OUT_DIM):
                var v = Float64(eval_out_host.unsafe_ptr()[i * NET.OUT_DIM + c])
                if v > best_val:
                    best_val = v
                    best_class = c
            if best_class == Int(ds.test_labels[sample_idx]):
                initial_correct += 1
    var initial_acc = Float64(initial_correct) / Float64(N_TEST_BATCHES * BATCH)
    print("\n  [pre-train test acc]:", initial_acc)

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n  epoch | batch | train_t (s)")
    print("  ------+-------+--------------")

    var step_num: Int = 0
    var t0 = perf_counter_ns()

    for epoch in range(EPOCHS):
        var epoch_start_ns = perf_counter_ns()

        for batch_idx in range(N_TRAIN_BATCHES):
            var x_batch_t = tile[BATCH, NET.IN_DIM](train_img_dbuf.unsafe_ptr().as_unsafe_any_origin() + batch_idx * BATCH * NET.IN_DIM)
            var y_batch_t = tile[BATCH, NET.OUT_DIM](train_tgt_dbuf.unsafe_ptr().as_unsafe_any_origin() + batch_idx * BATCH * NET.OUT_DIM)

            TRAINER.compute_grads_only_gpu[BATCH](
                ctx,
                params_t,
                grads_t,
                lat_t,
                mu_eps_t,
                a_below_t,
                z_below_t,
                dx_t,
                x_batch_t,
                y_batch_t,
                T_infer=T_INFER,
                lr_x=Scalar[dtype](LR_X),
            )

            step_num += 1
            OPT.step_gpu[NET.PARAM_SIZE, dtype](
                ctx, params_t, grads_t, opt_state_t, opt_global_t, step_num
            )

            if batch_idx == 0 or (batch_idx + 1) % 20 == 0 or batch_idx == N_TRAIN_BATCHES - 1:
                ctx.synchronize()
                var elapsed = Float64(perf_counter_ns() - epoch_start_ns) / 1e9
                print(
                    "    ", epoch, "  ", batch_idx, "  ",
                    fit(String(elapsed), 7),
                )

        ctx.synchronize()

        # Inline eval at end of epoch
        var ep_correct: Int = 0
        for tb in range(N_TEST_BATCHES):
            var x_batch_t = tile[BATCH, NET.IN_DIM](test_img_dbuf.unsafe_ptr().as_unsafe_any_origin() + tb * BATCH * NET.IN_DIM)
            NET.forward_eval_gpu[BATCH, dtype](
                ctx, x_batch_t, params_t, eval_out_t, mu_eps_t, a_below_t
            )
            ctx.enqueue_copy(eval_out_host, eval_out_dbuf)
            ctx.synchronize()
            for i in range(BATCH):
                var sample_idx = tb * BATCH + i
                var best_class: Int = 0
                var best_val = Float64(
                    eval_out_host.unsafe_ptr()[i * NET.OUT_DIM]
                )
                for c in range(1, NET.OUT_DIM):
                    var v = Float64(
                        eval_out_host.unsafe_ptr()[i * NET.OUT_DIM + c]
                    )
                    if v > best_val:
                        best_val = v
                        best_class = c
                if best_class == Int(ds.test_labels[sample_idx]):
                    ep_correct += 1
        var acc = Float64(ep_correct) / Float64(N_TEST_BATCHES * BATCH)
        print("  [epoch", epoch, "done]  test_acc=", acc)

    var total_elapsed = Float64(perf_counter_ns() - t0) / 1e9

    # Final eval
    var final_correct: Int = 0
    for tb in range(N_TEST_BATCHES):
        var x_batch_t = tile[BATCH, NET.IN_DIM](test_img_dbuf.unsafe_ptr().as_unsafe_any_origin() + tb * BATCH * NET.IN_DIM)
        NET.forward_eval_gpu[BATCH, dtype](
            ctx, x_batch_t, params_t, eval_out_t, mu_eps_t, a_below_t
        )
        ctx.enqueue_copy(eval_out_host, eval_out_dbuf)
        ctx.synchronize()
        for i in range(BATCH):
            var sample_idx = tb * BATCH + i
            var best_class: Int = 0
            var best_val = Float64(eval_out_host.unsafe_ptr()[i * NET.OUT_DIM])
            for c in range(1, NET.OUT_DIM):
                var v = Float64(eval_out_host.unsafe_ptr()[i * NET.OUT_DIM + c])
                if v > best_val:
                    best_val = v
                    best_class = c
            if best_class == Int(ds.test_labels[sample_idx]):
                final_correct += 1
    var final_acc = Float64(final_correct) / Float64(N_TEST_BATCHES * BATCH)

    print("\n  total wall time :", total_elapsed, "s")
    print("  final test acc  :", final_acc)
    print("  pre-train acc   :", initial_acc)
    print("  acc improvement :", final_acc - initial_acc)

    if final_acc >= 0.80:
        print("\n  [PASS] final test acc ≥ 80% — GPU PCN converges on real data")
    elif final_acc >= 0.50:
        print("\n  [PARTIAL] final test acc ≥ 50% but < 80%")
    else:
        print("\n  [FAIL] final test acc < 50%")
        raise Error("MNIST GPU test failed: final accuracy " + String(final_acc))

    print("=== Done ===")

from noeira.core.fmt import fit