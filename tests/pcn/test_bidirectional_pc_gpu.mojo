"""Bidirectional PC test (GPU port) — Bogacz notebook 5 reproduction.

GPU equivalent of test_bidirectional_pc.mojo. Same architecture, same
hyperparameters; all kernels run on the GPU. The custom joint-inference
logic uses two new kernels (defined inline here):

  dx_combine_apply: x_shared -= lr_x · (α_up·(ε_up_0 - z_up_1)
                                       + α_down·(ε_dn_0 - z_dn_1))
  grad_scale      : multiplies an array element-wise by a scalar
                    (used to scale DOWN grads by α_down before Adam)

All other PC ops use existing PCBlock GPU dispatchers.

Run:
    pixi run -e apple  mojo run -I . tests/pcn/test_bidirectional_pc_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/pcn/test_bidirectional_pc_gpu.mojo
"""

from std.memory import alloc, unsafe_memset
from std.time import perf_counter_ns
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from noeira.core.tiles import Tile, tile

from noeira.nn.constants import DT as dtype
from noeira.experimental.pcn.pc_constants import TPB
from noeira.experimental.pcn.pc_initializer import PCXavier
from noeira.experimental.pcn.pc_optimizer import PCAdam
from noeira.nn.datasets.mnist import MNIST
from noeira.experimental.pcn import (
    PCBlock,
    PCSequential,
    PCIdentity,
    PCReLU,
)


comptime BATCH = 100
comptime HIDDEN = 32
comptime EPOCHS = 3
comptime T_INFER = 15
comptime LR_X: Float64 = 0.01
comptime ADAM_LR: Float64 = 0.001
comptime ALPHA_UP: Float64 = 1.0
comptime ALPHA_DOWN: Float64 = 0.05

comptime N_TRAIN = 2000
comptime N_TEST = 500
comptime N_TRAIN_BATCHES = N_TRAIN // BATCH
comptime N_TEST_BATCHES = N_TEST // BATCH

comptime UB0 = PCBlock[784, HIDDEN, PCIdentity]
comptime UB1 = PCBlock[HIDDEN, 10, PCReLU]
comptime UP_NET = PCSequential[UB0, UB1]
comptime UP_PARAM_SIZE = UP_NET.PARAM_SIZE
comptime UB0_PARAM_SIZE = UB0.PARAM_SIZE
comptime UB1_PARAM_SIZE = UB1.PARAM_SIZE

comptime DB0 = PCBlock[10, HIDDEN, PCIdentity]
comptime DB1 = PCBlock[HIDDEN, 784, PCReLU]
comptime DOWN_NET = PCSequential[DB0, DB1]
comptime DOWN_PARAM_SIZE = DOWN_NET.PARAM_SIZE
comptime DB0_PARAM_SIZE = DB0.PARAM_SIZE
comptime DB1_PARAM_SIZE = DB1.PARAM_SIZE

comptime OPT = PCAdam[LR=ADAM_LR]


# =============================================================================
# Custom GPU kernels for the joint Phase D + DOWN grad scaling
# =============================================================================


def _dx_combine_apply_kernel[
    BATCH: Int,
    DIM: Int,
    KDT: DType,
](
    x_shared: Tile[KDT, BATCH, DIM],
    eps_up_0: Tile[KDT, BATCH, DIM],
    z_up_1: Tile[KDT, BATCH, DIM],
    eps_dn_0: Tile[KDT, BATCH, DIM],
    z_dn_1: Tile[KDT, BATCH, DIM],
    lr_x: Scalar[KDT],
    alpha_up: Scalar[KDT],
    alpha_down: Scalar[KDT],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * DIM:
        return
    var b = idx // DIM
    var k = idx % DIM
    var u = rebind[Scalar[KDT]](eps_up_0[b, k]) - rebind[Scalar[KDT]](
        z_up_1[b, k]
    )
    var d = rebind[Scalar[KDT]](eps_dn_0[b, k]) - rebind[Scalar[KDT]](
        z_dn_1[b, k]
    )
    x_shared[b, k] = rebind[Scalar[KDT]](x_shared[b, k]) - lr_x * (
        alpha_up * u + alpha_down * d
    )


def _scale_kernel[
    SIZE: Int,
    KDT: DType,
](
    arr: Tile[KDT, SIZE],
    factor: Scalar[KDT],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= SIZE:
        return
    arr[idx] = rebind[Scalar[KDT]](arr[idx]) * factor


def main() raises:
    print("=" * 60)
    print("Bidirectional PC (GPU) — Bogacz notebook 5 reproduction")
    print("=" * 60)
    print("  UP   arch  : 784 →", HIDDEN, "→ 10")
    print("  DOWN arch  : 10  →", HIDDEN, "→ 784")
    print(
        "  hyperparams: BATCH=", BATCH, " T_INFER=", T_INFER, " EPOCHS=", EPOCHS
    )
    print("  α_up=", ALPHA_UP, " α_down=", ALPHA_DOWN)

    var ctx = DeviceContext()
    var ds = MNIST()
    print("  [mnist] loaded")

    # ── UP params + Adam state on GPU ────────────────────────────────────────
    var up_params_host = ctx.enqueue_create_host_buffer[dtype](UP_PARAM_SIZE)
    for i in range(UP_PARAM_SIZE):
        up_params_host.unsafe_ptr()[i] = Scalar[dtype](0)
    var up_params_init_t = tile[UP_PARAM_SIZE](up_params_host.unsafe_ptr().as_unsafe_any_origin())
    UP_NET.pc_init_params[PCXavier, dtype](up_params_init_t)

    var up_params_dbuf = ctx.enqueue_create_buffer[dtype](UP_PARAM_SIZE)
    ctx.enqueue_copy(up_params_dbuf, up_params_host)
    var up_grads_dbuf = ctx.enqueue_create_buffer[dtype](UP_PARAM_SIZE)
    var up_opt_state_dbuf = ctx.enqueue_create_buffer[dtype](
        UP_PARAM_SIZE * OPT.STATE_PER_PARAM
    )
    var up_opt_global_dbuf = ctx.enqueue_create_buffer[dtype](
        OPT.GLOBAL_STATE_SIZE
    )
    var up_zero_state = ctx.enqueue_create_host_buffer[dtype](
        UP_PARAM_SIZE * OPT.STATE_PER_PARAM
    )
    for i in range(UP_PARAM_SIZE * OPT.STATE_PER_PARAM):
        up_zero_state.unsafe_ptr()[i] = Scalar[dtype](0)
    ctx.enqueue_copy(up_opt_state_dbuf, up_zero_state)
    var up_opt_global_init = ctx.enqueue_create_host_buffer[dtype](
        OPT.GLOBAL_STATE_SIZE
    )
    up_opt_global_init.unsafe_ptr()[0] = Scalar[dtype](0)
    up_opt_global_init.unsafe_ptr()[1] = Scalar[dtype](1.0)
    ctx.enqueue_copy(up_opt_global_dbuf, up_opt_global_init)

    # ── DOWN params + Adam state on GPU ──────────────────────────────────────
    var dn_params_host = ctx.enqueue_create_host_buffer[dtype](DOWN_PARAM_SIZE)
    for i in range(DOWN_PARAM_SIZE):
        dn_params_host.unsafe_ptr()[i] = Scalar[dtype](0)
    var dn_params_init_t = tile[DOWN_PARAM_SIZE](dn_params_host.unsafe_ptr().as_unsafe_any_origin())
    DOWN_NET.pc_init_params[PCXavier, dtype](dn_params_init_t)

    var dn_params_dbuf = ctx.enqueue_create_buffer[dtype](DOWN_PARAM_SIZE)
    ctx.enqueue_copy(dn_params_dbuf, dn_params_host)
    var dn_grads_dbuf = ctx.enqueue_create_buffer[dtype](DOWN_PARAM_SIZE)
    var dn_opt_state_dbuf = ctx.enqueue_create_buffer[dtype](
        DOWN_PARAM_SIZE * OPT.STATE_PER_PARAM
    )
    var dn_opt_global_dbuf = ctx.enqueue_create_buffer[dtype](
        OPT.GLOBAL_STATE_SIZE
    )
    var dn_zero_state = ctx.enqueue_create_host_buffer[dtype](
        DOWN_PARAM_SIZE * OPT.STATE_PER_PARAM
    )
    for i in range(DOWN_PARAM_SIZE * OPT.STATE_PER_PARAM):
        dn_zero_state.unsafe_ptr()[i] = Scalar[dtype](0)
    ctx.enqueue_copy(dn_opt_state_dbuf, dn_zero_state)
    var dn_opt_global_init = ctx.enqueue_create_host_buffer[dtype](
        OPT.GLOBAL_STATE_SIZE
    )
    dn_opt_global_init.unsafe_ptr()[0] = Scalar[dtype](0)
    dn_opt_global_init.unsafe_ptr()[1] = Scalar[dtype](1.0)
    ctx.enqueue_copy(dn_opt_global_dbuf, dn_opt_global_init)

    # Param/grad views
    var up_params_t = tile[UP_PARAM_SIZE](up_params_dbuf)
    var up_grads_t = tile[UP_PARAM_SIZE](up_grads_dbuf)
    var up_opt_state_t = tile[UP_PARAM_SIZE, OPT.STATE_PER_PARAM](up_opt_state_dbuf)
    var up_opt_global_t = tile[OPT.GLOBAL_STATE_SIZE](up_opt_global_dbuf)

    var dn_params_t = tile[DOWN_PARAM_SIZE](dn_params_dbuf)
    var dn_grads_t = tile[DOWN_PARAM_SIZE](dn_grads_dbuf)
    var dn_opt_state_t = tile[DOWN_PARAM_SIZE, OPT.STATE_PER_PARAM](dn_opt_state_dbuf)
    var dn_opt_global_t = tile[OPT.GLOBAL_STATE_SIZE](dn_opt_global_dbuf)

    # Per-block param sub-views (for individual PCBlock GPU calls)
    var up_p0_t = tile[UB0_PARAM_SIZE](up_params_dbuf.unsafe_ptr().as_unsafe_any_origin())
    var up_p1_t = tile[UB1_PARAM_SIZE](up_params_dbuf.unsafe_ptr().as_unsafe_any_origin() + UB0_PARAM_SIZE)
    var up_g0_t = tile[UB0_PARAM_SIZE](up_grads_dbuf.unsafe_ptr().as_unsafe_any_origin())
    var up_g1_t = tile[UB1_PARAM_SIZE](up_grads_dbuf.unsafe_ptr().as_unsafe_any_origin() + UB0_PARAM_SIZE)
    var dn_p0_t = tile[DB0_PARAM_SIZE](dn_params_dbuf.unsafe_ptr().as_unsafe_any_origin())
    var dn_p1_t = tile[DB1_PARAM_SIZE](dn_params_dbuf.unsafe_ptr().as_unsafe_any_origin() + DB0_PARAM_SIZE)
    var dn_g0_t = tile[DB0_PARAM_SIZE](dn_grads_dbuf.unsafe_ptr().as_unsafe_any_origin())
    var dn_g1_t = tile[DB1_PARAM_SIZE](dn_grads_dbuf.unsafe_ptr().as_unsafe_any_origin() + DB0_PARAM_SIZE)

    # ── Shared latent (GPU) ──────────────────────────────────────────────────
    var x_shared_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    var x_shared_t = tile[BATCH, HIDDEN](x_shared_dbuf)

    # ── Per-path scratch buffers (GPU) ───────────────────────────────────────
    var up_mu0_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    var up_eps0_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    var up_mu1_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * 10)
    var up_eps1_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * 10)
    var up_a0_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * 784)
    var up_a1_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    var up_z1_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)

    var dn_mu0_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    var dn_eps0_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    var dn_mu1_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * 784)
    var dn_eps1_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * 784)
    var dn_a0_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * 10)
    var dn_a1_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    var dn_z1_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)

    var up_mu0_t = tile[BATCH, HIDDEN](up_mu0_dbuf)
    var up_eps0_t = tile[BATCH, HIDDEN](up_eps0_dbuf)
    var up_mu1_t = tile[BATCH, 10](up_mu1_dbuf)
    var up_eps1_t = tile[BATCH, 10](up_eps1_dbuf)
    var up_a0_t = tile[BATCH, 784](up_a0_dbuf)
    var up_a1_t = tile[BATCH, HIDDEN](up_a1_dbuf)
    var up_z1_t = tile[BATCH, HIDDEN](up_z1_dbuf)
    var dn_mu0_t = tile[BATCH, HIDDEN](dn_mu0_dbuf)
    var dn_eps0_t = tile[BATCH, HIDDEN](dn_eps0_dbuf)
    var dn_mu1_t = tile[BATCH, 784](dn_mu1_dbuf)
    var dn_eps1_t = tile[BATCH, 784](dn_eps1_dbuf)
    var dn_a0_t = tile[BATCH, 10](dn_a0_dbuf)
    var dn_a1_t = tile[BATCH, HIDDEN](dn_a1_dbuf)
    var dn_z1_t = tile[BATCH, HIDDEN](dn_z1_dbuf)

    # ── Upload entire MNIST train + test to GPU ──────────────────────────────
    var train_img_host = ctx.enqueue_create_host_buffer[dtype](
        MNIST.N_TRAIN * 784
    )
    var train_lbl_host = ctx.enqueue_create_host_buffer[dtype](
        MNIST.N_TRAIN * 10
    )
    for i in range(MNIST.N_TRAIN * 784):
        train_img_host.unsafe_ptr()[i] = ds.train_images[i]
    for i in range(MNIST.N_TRAIN * 10):
        train_lbl_host.unsafe_ptr()[i] = Scalar[dtype](0)
    for i in range(MNIST.N_TRAIN):
        train_lbl_host.unsafe_ptr()[i * 10 + Int(ds.train_labels[i])] = Scalar[
            dtype
        ](1.0)
    var train_img_dbuf = ctx.enqueue_create_buffer[dtype](MNIST.N_TRAIN * 784)
    var train_lbl_dbuf = ctx.enqueue_create_buffer[dtype](MNIST.N_TRAIN * 10)
    ctx.enqueue_copy(train_img_dbuf, train_img_host)
    ctx.enqueue_copy(train_lbl_dbuf, train_lbl_host)

    var test_img_host = ctx.enqueue_create_host_buffer[dtype](
        MNIST.N_TEST * 784
    )
    for i in range(MNIST.N_TEST * 784):
        test_img_host.unsafe_ptr()[i] = ds.test_images[i]
    var test_img_dbuf = ctx.enqueue_create_buffer[dtype](MNIST.N_TEST * 784)
    ctx.enqueue_copy(test_img_dbuf, test_img_host)

    ctx.synchronize()
    print("  [gpu] datasets uploaded")

    # ── Train ────────────────────────────────────────────────────────────────
    print("\n  epoch | wall_t (s)")
    print("  ------+------------")
    var step_num: Int = 0
    var t0 = perf_counter_ns()

    var lr_x_s = Scalar[dtype](LR_X)
    var alpha_up_s = Scalar[dtype](ALPHA_UP)
    var alpha_down_s = Scalar[dtype](ALPHA_DOWN)
    comptime k_dx = _dx_combine_apply_kernel[BATCH, HIDDEN, dtype]
    comptime k_scale = _scale_kernel[DOWN_PARAM_SIZE, dtype]
    var dx_threads = BATCH * HIDDEN
    var dx_blocks = (dx_threads + TPB - 1) // TPB
    var scale_threads = DOWN_PARAM_SIZE
    var scale_blocks = (scale_threads + TPB - 1) // TPB

    for epoch in range(EPOCHS):
        for batch_idx in range(N_TRAIN_BATCHES):
            # Per-batch input views into pre-uploaded MNIST data
            var image_t = tile[BATCH, 784](train_img_dbuf.unsafe_ptr().as_unsafe_any_origin() + batch_idx * BATCH * 784)
            var label_oh_t = tile[BATCH, 10](train_lbl_dbuf.unsafe_ptr().as_unsafe_any_origin() + batch_idx * BATCH * 10)

            # Init x_shared via UP forward sweep: x_shared = μ_up_0
            UB0.predict_gpu[BATCH, dtype](
                ctx, image_t, up_p0_t, up_mu0_t, up_a0_t
            )
            ctx.enqueue_copy(x_shared_dbuf, up_mu0_dbuf)

            # T_INFER iterations of joint inference
            for _ in range(T_INFER):
                # UP path: predict + ε
                UB0.predict_gpu[BATCH, dtype](
                    ctx, image_t, up_p0_t, up_mu0_t, up_a0_t
                )
                UB1.predict_gpu[BATCH, dtype](
                    ctx, x_shared_t, up_p1_t, up_mu1_t, up_a1_t
                )
                UB0.eps_compute_gpu[BATCH, dtype](
                    ctx, x_shared_t, up_mu0_t, up_eps0_t
                )
                UB1.eps_compute_gpu[BATCH, dtype](
                    ctx, label_oh_t, up_mu1_t, up_eps1_t
                )

                # DOWN path: predict + ε
                DB0.predict_gpu[BATCH, dtype](
                    ctx, label_oh_t, dn_p0_t, dn_mu0_t, dn_a0_t
                )
                DB1.predict_gpu[BATCH, dtype](
                    ctx, x_shared_t, dn_p1_t, dn_mu1_t, dn_a1_t
                )
                DB0.eps_compute_gpu[BATCH, dtype](
                    ctx, x_shared_t, dn_mu0_t, dn_eps0_t
                )
                DB1.eps_compute_gpu[BATCH, dtype](
                    ctx, image_t, dn_mu1_t, dn_eps1_t
                )

                # Phase C: pull-backs gated by act'(x_shared)
                UB1.pull_back_gpu[BATCH, dtype](
                    ctx, up_eps1_t, up_p1_t, up_z1_t
                )
                UB1.act_derivative_mul_gpu[BATCH, dtype](
                    ctx, x_shared_t, up_z1_t, up_z1_t
                )
                DB1.pull_back_gpu[BATCH, dtype](
                    ctx, dn_eps1_t, dn_p1_t, dn_z1_t
                )
                DB1.act_derivative_mul_gpu[BATCH, dtype](
                    ctx, x_shared_t, dn_z1_t, dn_z1_t
                )

                # Phase D: x_shared -= lr_x · (α_up·(ε_up_0 - z_up_1)
                #                              + α_down·(ε_dn_0 - z_dn_1))
                ctx.enqueue_function[k_dx](
                    x_shared_t,
                    up_eps0_t,
                    up_z1_t,
                    dn_eps0_t,
                    dn_z1_t,
                    lr_x_s,
                    alpha_up_s,
                    alpha_down_s,
                    grid_dim=(dx_blocks,),
                    block_dim=(TPB,),
                )

            # Weight grads (using post-inference ε)
            UB0.weight_grad_gpu[BATCH, dtype](ctx, up_eps0_t, up_a0_t, up_g0_t)
            UB1.weight_grad_gpu[BATCH, dtype](ctx, up_eps1_t, up_a1_t, up_g1_t)
            DB0.weight_grad_gpu[BATCH, dtype](ctx, dn_eps0_t, dn_a0_t, dn_g0_t)
            DB1.weight_grad_gpu[BATCH, dtype](ctx, dn_eps1_t, dn_a1_t, dn_g1_t)

            # Scale DOWN grads by alpha_down
            ctx.enqueue_function[k_scale](
                dn_grads_t,
                alpha_down_s,
                grid_dim=(scale_blocks,),
                block_dim=(TPB,),
            )

            # Adam steps
            step_num += 1
            OPT.step_gpu[UP_PARAM_SIZE, dtype](
                ctx,
                up_params_t,
                up_grads_t,
                up_opt_state_t,
                up_opt_global_t,
                step_num,
            )
            OPT.step_gpu[DOWN_PARAM_SIZE, dtype](
                ctx,
                dn_params_t,
                dn_grads_t,
                dn_opt_state_t,
                dn_opt_global_t,
                step_num,
            )

        ctx.synchronize()
        var elapsed = Float64(perf_counter_ns() - t0) / 1e9
        print("    ", epoch, "  ", elapsed)

    ctx.synchronize()
    var total_t = Float64(perf_counter_ns() - t0) / 1e9
    print("\n  total train time:", total_t, "s")

    # ── Eval: classification accuracy via UP forward_eval_gpu ────────────────
    var pred_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * 10)
    var pred_t = tile[BATCH, 10](pred_dbuf)
    var up_eval_mu_dbuf = ctx.enqueue_create_buffer[dtype](
        BATCH * UP_NET.SCRATCH_OUT_DIM
    )
    var up_eval_a_dbuf = ctx.enqueue_create_buffer[dtype](
        BATCH * UP_NET.SCRATCH_IN_DIM
    )
    var up_eval_mu_t = tile[BATCH, UP_NET.SCRATCH_OUT_DIM](up_eval_mu_dbuf)
    var up_eval_a_t = tile[BATCH, UP_NET.SCRATCH_IN_DIM](up_eval_a_dbuf)
    var pred_host = ctx.enqueue_create_host_buffer[dtype](BATCH * 10)

    var correct: Int = 0
    for tb in range(N_TEST_BATCHES):
        var test_img_t = tile[BATCH, 784](test_img_dbuf.unsafe_ptr().as_unsafe_any_origin() + tb * BATCH * 784)
        UP_NET.forward_eval_gpu[BATCH, dtype](
            ctx, test_img_t, up_params_t, pred_t, up_eval_mu_t, up_eval_a_t
        )
        ctx.enqueue_copy(pred_host, pred_dbuf)
        ctx.synchronize()
        for i in range(BATCH):
            var sample_idx = tb * BATCH + i
            var best_class: Int = 0
            var best_val = Float64(pred_host.unsafe_ptr()[i * 10])
            for c in range(1, 10):
                var v = Float64(pred_host.unsafe_ptr()[i * 10 + c])
                if v > best_val:
                    best_val = v
                    best_class = c
            if best_class == Int(ds.test_labels[sample_idx]):
                correct += 1
    var test_acc = Float64(correct) / Float64(N_TEST_BATCHES * BATCH)
    print("\n  UP test accuracy:", test_acc)

    # ── Generation: forward_eval_gpu on DOWN with one-hot labels ─────────────
    var gen_label_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * 10)
    var gen_image_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * 784)
    var dn_eval_mu_dbuf = ctx.enqueue_create_buffer[dtype](
        BATCH * DOWN_NET.SCRATCH_OUT_DIM
    )
    var dn_eval_a_dbuf = ctx.enqueue_create_buffer[dtype](
        BATCH * DOWN_NET.SCRATCH_IN_DIM
    )
    var gen_label_host = ctx.enqueue_create_host_buffer[dtype](BATCH * 10)
    for i in range(BATCH * 10):
        gen_label_host.unsafe_ptr()[i] = Scalar[dtype](0)
    for c in range(10):
        gen_label_host.unsafe_ptr()[c * 10 + c] = Scalar[dtype](1.0)
    ctx.enqueue_copy(gen_label_dbuf, gen_label_host)
    var gen_label_t = tile[BATCH, 10](gen_label_dbuf)
    var gen_image_t = tile[BATCH, 784](gen_image_dbuf)
    var dn_eval_mu_t = tile[BATCH, DOWN_NET.SCRATCH_OUT_DIM](dn_eval_mu_dbuf)
    var dn_eval_a_t = tile[BATCH, DOWN_NET.SCRATCH_IN_DIM](dn_eval_a_dbuf)
    DOWN_NET.forward_eval_gpu[BATCH, dtype](
        ctx, gen_label_t, dn_params_t, gen_image_t, dn_eval_mu_t, dn_eval_a_t
    )
    var gen_image_host = ctx.enqueue_create_host_buffer[dtype](BATCH * 784)
    ctx.enqueue_copy(gen_image_host, gen_image_dbuf)
    ctx.synchronize()

    var class_diff: Float64 = 0
    for j in range(784):
        var d = Float64(gen_image_host.unsafe_ptr()[0 * 784 + j]) - Float64(
            gen_image_host.unsafe_ptr()[9 * 784 + j]
        )
        class_diff += d * d
    class_diff /= Float64(784)

    var gen_mean_mag: Float64 = 0
    for c in range(10):
        for j in range(784):
            gen_mean_mag += abs(
                Float64(gen_image_host.unsafe_ptr()[c * 784 + j])
            )
    gen_mean_mag /= Float64(10 * 784)

    print("  Per-pixel MSE between class-0 and class-9:", class_diff)
    print("  Mean |gen pixel| across 10 classes        :", gen_mean_mag)

    if test_acc >= 0.50:
        print("\n  [PASS] bidirectional PC GPU: classifier reaches", test_acc)
        if class_diff < 0.005:
            print(
                "  (NOTE) decoder under-trained (matches CPU result — α_down"
                " low or budget short)"
            )
    else:
        print("\n  [FAIL] classification too low")
        raise Error("bidirectional GPU test failed")

    print("=== Done ===")
