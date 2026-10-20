"""Bidirectional PC (GPU) — full Bogacz notebook 5 reproduction.

Mirrors `references/pcn/PredictiveCoding-main/5_bidirectional_pc.ipynb`
exactly:
  - UP   : 784 → 256 → x_1 → 256 → x_2 → 10
  - DOWN : 10  → 256 → x_2 → 256 → x_1 → 784   (latents x_1, x_2 SHARED with up)
  - α_up=1.0, α_down=0.0001 (gen is auxiliary regularizer in their setup)
  - T=20 inference iterations, batch_size=500, 10000 train, 1000 test, 10 epochs
  - lr_x=0.01 (latent SGD), Adam lr=1e-2 (note: 10x notebook 1's Adam lr)

Run:
    pixi run -e apple  mojo run -I . tests/pcn/test_bidirectional_pc_gpu_full.mojo
    pixi run -e nvidia mojo run -I . tests/pcn/test_bidirectional_pc_gpu_full.mojo
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


# ── Bogacz notebook 5 hyperparameters ────────────────────────────────────────
comptime BATCH = 500
comptime HIDDEN = 256
comptime EPOCHS = 10
comptime T_INFER = 20
comptime LR_X: Float64 = 0.01
comptime ADAM_LR: Float64 = 1.0e-2  # 10x notebook 1's lr
comptime ALPHA_UP: Float64 = 1.0
comptime ALPHA_DOWN: Float64 = 1.0e-4

comptime N_TRAIN = 10000
comptime N_TEST = 1000
comptime N_TRAIN_BATCHES = N_TRAIN // BATCH  # 20
comptime N_TEST_BATCHES = N_TEST // BATCH  # 2

# UP path: 3 blocks
comptime UB0 = PCBlock[784, HIDDEN, PCIdentity]  # image → x_1
comptime UB1 = PCBlock[HIDDEN, HIDDEN, PCReLU]  # ReLU(x_1) → x_2
comptime UB2 = PCBlock[HIDDEN, 10, PCReLU]  # ReLU(x_2) → label
comptime UP_NET = PCSequential[UB0, UB1, UB2]
comptime UP_PARAM_SIZE = UP_NET.PARAM_SIZE
comptime UB0_PARAM_SIZE = UB0.PARAM_SIZE
comptime UB1_PARAM_SIZE = UB1.PARAM_SIZE
comptime UB2_PARAM_SIZE = UB2.PARAM_SIZE

# DOWN path: 3 blocks (latents x_2, then x_1 — REVERSED relative to up)
comptime DB0 = PCBlock[10, HIDDEN, PCIdentity]  # label → x_2
comptime DB1 = PCBlock[HIDDEN, HIDDEN, PCReLU]  # ReLU(x_2) → x_1
comptime DB2 = PCBlock[HIDDEN, 784, PCReLU]  # ReLU(x_1) → image
comptime DOWN_NET = PCSequential[DB0, DB1, DB2]
comptime DOWN_PARAM_SIZE = DOWN_NET.PARAM_SIZE
comptime DB0_PARAM_SIZE = DB0.PARAM_SIZE
comptime DB1_PARAM_SIZE = DB1.PARAM_SIZE
comptime DB2_PARAM_SIZE = DB2.PARAM_SIZE

comptime OPT = PCAdam[LR=ADAM_LR]


# =============================================================================
# Reusable kernels (same as small GPU test)
# =============================================================================


def _dx_combine_apply_kernel[
    BATCH: Int,
    DIM: Int,
    KDT: DType,
](
    x_lat: Tile[KDT, BATCH, DIM],
    eps_self_up: Tile[KDT, BATCH, DIM],
    z_above_up: Tile[KDT, BATCH, DIM],
    eps_self_dn: Tile[KDT, BATCH, DIM],
    z_above_dn: Tile[KDT, BATCH, DIM],
    lr_x: Scalar[KDT],
    alpha_up: Scalar[KDT],
    alpha_down: Scalar[KDT],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * DIM:
        return
    var b = idx // DIM
    var k = idx % DIM
    var u = rebind[Scalar[KDT]](eps_self_up[b, k]) - rebind[Scalar[KDT]](
        z_above_up[b, k]
    )
    var d = rebind[Scalar[KDT]](eps_self_dn[b, k]) - rebind[Scalar[KDT]](
        z_above_dn[b, k]
    )
    x_lat[b, k] = rebind[Scalar[KDT]](x_lat[b, k]) - lr_x * (
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


def _copy_kernel[
    SIZE: Int,
    KDT: DType,
](
    src: Tile[KDT, SIZE],
    dst: Tile[KDT, SIZE],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= SIZE:
        return
    dst[idx] = rebind[Scalar[KDT]](src[idx])


def main() raises:
    print("=" * 60)
    print("Bidirectional PC (GPU, full notebook 5) — Bogacz reproduction")
    print("=" * 60)
    print("  UP   arch  : 784 → 256 → x_1 → 256 → x_2 → 10")
    print("  DOWN arch  : 10  → 256 → x_2 → 256 → x_1 → 784  (latents shared)")
    print(
        "  HIDDEN=",
        HIDDEN,
        " BATCH=",
        BATCH,
        " T_INFER=",
        T_INFER,
        " EPOCHS=",
        EPOCHS,
    )
    print("  α_up=", ALPHA_UP, " α_down=", ALPHA_DOWN, " Adam lr=", ADAM_LR)
    print("  N_TRAIN=", N_TRAIN, " N_TEST=", N_TEST)

    var ctx = DeviceContext()
    var ds = MNIST()
    print("  [mnist] loaded")

    # ── UP/DOWN params + grads + Adam state on GPU ───────────────────────────
    var up_params_host = ctx.enqueue_create_host_buffer[dtype](UP_PARAM_SIZE)
    for i in range(UP_PARAM_SIZE):
        up_params_host.unsafe_ptr()[i] = Scalar[dtype](0)
    var up_init_t = tile[UP_PARAM_SIZE](up_params_host.unsafe_ptr().as_unsafe_any_origin())
    UP_NET.pc_init_params[PCXavier, dtype](up_init_t)

    var up_params_dbuf = ctx.enqueue_create_buffer[dtype](UP_PARAM_SIZE)
    var up_grads_dbuf = ctx.enqueue_create_buffer[dtype](UP_PARAM_SIZE)
    var up_opt_state_dbuf = ctx.enqueue_create_buffer[dtype](
        UP_PARAM_SIZE * OPT.STATE_PER_PARAM
    )
    var up_opt_global_dbuf = ctx.enqueue_create_buffer[dtype](
        OPT.GLOBAL_STATE_SIZE
    )
    ctx.enqueue_copy(up_params_dbuf, up_params_host)
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

    var dn_params_host = ctx.enqueue_create_host_buffer[dtype](DOWN_PARAM_SIZE)
    for i in range(DOWN_PARAM_SIZE):
        dn_params_host.unsafe_ptr()[i] = Scalar[dtype](0)
    var dn_init_t = tile[DOWN_PARAM_SIZE](dn_params_host.unsafe_ptr().as_unsafe_any_origin())
    DOWN_NET.pc_init_params[PCXavier, dtype](dn_init_t)

    var dn_params_dbuf = ctx.enqueue_create_buffer[dtype](DOWN_PARAM_SIZE)
    var dn_grads_dbuf = ctx.enqueue_create_buffer[dtype](DOWN_PARAM_SIZE)
    var dn_opt_state_dbuf = ctx.enqueue_create_buffer[dtype](
        DOWN_PARAM_SIZE * OPT.STATE_PER_PARAM
    )
    var dn_opt_global_dbuf = ctx.enqueue_create_buffer[dtype](
        OPT.GLOBAL_STATE_SIZE
    )
    ctx.enqueue_copy(dn_params_dbuf, dn_params_host)
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

    var up_params_t = tile[UP_PARAM_SIZE](up_params_dbuf)
    var up_grads_t = tile[UP_PARAM_SIZE](up_grads_dbuf)
    var up_opt_state_t = tile[UP_PARAM_SIZE, OPT.STATE_PER_PARAM](up_opt_state_dbuf)
    var up_opt_global_t = tile[OPT.GLOBAL_STATE_SIZE](up_opt_global_dbuf)

    var dn_params_t = tile[DOWN_PARAM_SIZE](dn_params_dbuf)
    var dn_grads_t = tile[DOWN_PARAM_SIZE](dn_grads_dbuf)
    var dn_opt_state_t = tile[DOWN_PARAM_SIZE, OPT.STATE_PER_PARAM](dn_opt_state_dbuf)
    var dn_opt_global_t = tile[OPT.GLOBAL_STATE_SIZE](dn_opt_global_dbuf)

    # Per-block param sub-views
    var up_p0_t = tile[UB0_PARAM_SIZE](up_params_dbuf.unsafe_ptr().as_unsafe_any_origin())
    var up_p1_t = tile[UB1_PARAM_SIZE](up_params_dbuf.unsafe_ptr().as_unsafe_any_origin() + UB0_PARAM_SIZE)
    var up_p2_t = tile[UB2_PARAM_SIZE](up_params_dbuf.unsafe_ptr().as_unsafe_any_origin() + UB0_PARAM_SIZE + UB1_PARAM_SIZE)
    var up_g0_t = tile[UB0_PARAM_SIZE](up_grads_dbuf.unsafe_ptr().as_unsafe_any_origin())
    var up_g1_t = tile[UB1_PARAM_SIZE](up_grads_dbuf.unsafe_ptr().as_unsafe_any_origin() + UB0_PARAM_SIZE)
    var up_g2_t = tile[UB2_PARAM_SIZE](up_grads_dbuf.unsafe_ptr().as_unsafe_any_origin() + UB0_PARAM_SIZE + UB1_PARAM_SIZE)

    var dn_p0_t = tile[DB0_PARAM_SIZE](dn_params_dbuf.unsafe_ptr().as_unsafe_any_origin())
    var dn_p1_t = tile[DB1_PARAM_SIZE](dn_params_dbuf.unsafe_ptr().as_unsafe_any_origin() + DB0_PARAM_SIZE)
    var dn_p2_t = tile[DB2_PARAM_SIZE](dn_params_dbuf.unsafe_ptr().as_unsafe_any_origin() + DB0_PARAM_SIZE + DB1_PARAM_SIZE)
    var dn_g0_t = tile[DB0_PARAM_SIZE](dn_grads_dbuf.unsafe_ptr().as_unsafe_any_origin())
    var dn_g1_t = tile[DB1_PARAM_SIZE](dn_grads_dbuf.unsafe_ptr().as_unsafe_any_origin() + DB0_PARAM_SIZE)
    var dn_g2_t = tile[DB2_PARAM_SIZE](dn_grads_dbuf.unsafe_ptr().as_unsafe_any_origin() + DB0_PARAM_SIZE + DB1_PARAM_SIZE)

    # ── Two SHARED latents: x_1, x_2 (each [BATCH, HIDDEN]) ──────────────────
    var x1_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    var x2_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    var x1_t = tile[BATCH, HIDDEN](x1_dbuf)
    var x2_t = tile[BATCH, HIDDEN](x2_dbuf)

    # ── Per-block scratch buffers (μ, ε, a_below, z_below) ───────────────────
    # UP block_0: 784 → HIDDEN
    var u_mu0 = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    var u_eps0 = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    var u_a0 = ctx.enqueue_create_buffer[dtype](BATCH * 784)
    # UP block_1: HIDDEN → HIDDEN
    var u_mu1 = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    var u_eps1 = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    var u_a1 = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    var u_z1 = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    # UP block_2: HIDDEN → 10
    var u_mu2 = ctx.enqueue_create_buffer[dtype](BATCH * 10)
    var u_eps2 = ctx.enqueue_create_buffer[dtype](BATCH * 10)
    var u_a2 = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    var u_z2 = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)

    # DOWN block_0: 10 → HIDDEN
    var d_mu0 = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    var d_eps0 = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    var d_a0 = ctx.enqueue_create_buffer[dtype](BATCH * 10)
    # DOWN block_1: HIDDEN → HIDDEN
    var d_mu1 = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    var d_eps1 = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    var d_a1 = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    var d_z1 = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    # DOWN block_2: HIDDEN → 784
    var d_mu2 = ctx.enqueue_create_buffer[dtype](BATCH * 784)
    var d_eps2 = ctx.enqueue_create_buffer[dtype](BATCH * 784)
    var d_a2 = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    var d_z2 = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)

    # Tensor views for everything we use in the loop
    var u_mu0_t = tile[BATCH, HIDDEN](u_mu0)
    var u_eps0_t = tile[BATCH, HIDDEN](u_eps0)
    var u_a0_t = tile[BATCH, 784](u_a0)
    var u_mu1_t = tile[BATCH, HIDDEN](u_mu1)
    var u_eps1_t = tile[BATCH, HIDDEN](u_eps1)
    var u_a1_t = tile[BATCH, HIDDEN](u_a1)
    var u_z1_t = tile[BATCH, HIDDEN](u_z1)
    var u_mu2_t = tile[BATCH, 10](u_mu2)
    var u_eps2_t = tile[BATCH, 10](u_eps2)
    var u_a2_t = tile[BATCH, HIDDEN](u_a2)
    var u_z2_t = tile[BATCH, HIDDEN](u_z2)

    var d_mu0_t = tile[BATCH, HIDDEN](d_mu0)
    var d_eps0_t = tile[BATCH, HIDDEN](d_eps0)
    var d_a0_t = tile[BATCH, 10](d_a0)
    var d_mu1_t = tile[BATCH, HIDDEN](d_mu1)
    var d_eps1_t = tile[BATCH, HIDDEN](d_eps1)
    var d_a1_t = tile[BATCH, HIDDEN](d_a1)
    var d_z1_t = tile[BATCH, HIDDEN](d_z1)
    var d_mu2_t = tile[BATCH, 784](d_mu2)
    var d_eps2_t = tile[BATCH, 784](d_eps2)
    var d_a2_t = tile[BATCH, HIDDEN](d_a2)
    var d_z2_t = tile[BATCH, HIDDEN](d_z2)

    # ── Upload MNIST train + test sets ───────────────────────────────────────
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
    comptime k_copy = _copy_kernel[BATCH * HIDDEN, dtype]
    var dx_threads = BATCH * HIDDEN
    var dx_blocks = (dx_threads + TPB - 1) // TPB
    var scale_threads = DOWN_PARAM_SIZE
    var scale_blocks = (scale_threads + TPB - 1) // TPB
    var copy_threads = BATCH * HIDDEN
    var copy_blocks = (copy_threads + TPB - 1) // TPB

    # Flat views over latents for the copy kernel
    var x1_flat_t = tile[BATCH * HIDDEN](x1_dbuf)
    var x2_flat_t = tile[BATCH * HIDDEN](x2_dbuf)
    var u_mu0_flat_t = tile[BATCH * HIDDEN](u_mu0)
    var u_mu1_flat_t = tile[BATCH * HIDDEN](u_mu1)

    for epoch in range(EPOCHS):
        for batch_idx in range(N_TRAIN_BATCHES):
            var image_t = tile[BATCH, 784](train_img_dbuf.unsafe_ptr().as_unsafe_any_origin() + batch_idx * BATCH * 784)
            var label_oh_t = tile[BATCH, 10](train_lbl_dbuf.unsafe_ptr().as_unsafe_any_origin() + batch_idx * BATCH * 10)

            # Init latents via UP forward sweep:
            #   x_1 ← μ_up_0 = W_up_0·image + b
            #   x_2 ← μ_up_1 = W_up_1·ReLU(x_1) + b
            UB0.predict_gpu[BATCH, dtype](
                ctx, image_t, up_p0_t, u_mu0_t, u_a0_t
            )
            ctx.enqueue_function[k_copy](
                u_mu0_flat_t,
                x1_flat_t,
                grid_dim=(copy_blocks,),
                block_dim=(TPB,),
            )
            UB1.predict_gpu[BATCH, dtype](ctx, x1_t, up_p1_t, u_mu1_t, u_a1_t)
            ctx.enqueue_function[k_copy](
                u_mu1_flat_t,
                x2_flat_t,
                grid_dim=(copy_blocks,),
                block_dim=(TPB,),
            )

            # T_INFER iterations of joint inference
            for _ in range(T_INFER):
                # ---- UP forward + ε ------------------------------------------
                UB0.predict_gpu[BATCH, dtype](
                    ctx, image_t, up_p0_t, u_mu0_t, u_a0_t
                )
                UB1.predict_gpu[BATCH, dtype](
                    ctx, x1_t, up_p1_t, u_mu1_t, u_a1_t
                )
                UB2.predict_gpu[BATCH, dtype](
                    ctx, x2_t, up_p2_t, u_mu2_t, u_a2_t
                )
                UB0.eps_compute_gpu[BATCH, dtype](ctx, x1_t, u_mu0_t, u_eps0_t)
                UB1.eps_compute_gpu[BATCH, dtype](ctx, x2_t, u_mu1_t, u_eps1_t)
                UB2.eps_compute_gpu[BATCH, dtype](
                    ctx, label_oh_t, u_mu2_t, u_eps2_t
                )

                # ---- DOWN forward + ε ----------------------------------------
                DB0.predict_gpu[BATCH, dtype](
                    ctx, label_oh_t, dn_p0_t, d_mu0_t, d_a0_t
                )
                DB1.predict_gpu[BATCH, dtype](
                    ctx, x2_t, dn_p1_t, d_mu1_t, d_a1_t
                )
                DB2.predict_gpu[BATCH, dtype](
                    ctx, x1_t, dn_p2_t, d_mu2_t, d_a2_t
                )
                DB0.eps_compute_gpu[BATCH, dtype](ctx, x2_t, d_mu0_t, d_eps0_t)
                DB1.eps_compute_gpu[BATCH, dtype](ctx, x1_t, d_mu1_t, d_eps1_t)
                DB2.eps_compute_gpu[BATCH, dtype](
                    ctx, image_t, d_mu2_t, d_eps2_t
                )

                # ---- Phase C contributions for x_1 ---------------------------
                # UP: pull_back ε_up_1 through W_up_1 → u_z1; gate by act'(x_1)
                UB1.pull_back_gpu[BATCH, dtype](ctx, u_eps1_t, up_p1_t, u_z1_t)
                UB1.act_derivative_mul_gpu[BATCH, dtype](
                    ctx, x1_t, u_z1_t, u_z1_t
                )
                # DOWN: pull_back ε_dn_2 through W_dn_2 → d_z2; gate by act'(x_1)
                DB2.pull_back_gpu[BATCH, dtype](ctx, d_eps2_t, dn_p2_t, d_z2_t)
                DB2.act_derivative_mul_gpu[BATCH, dtype](
                    ctx, x1_t, d_z2_t, d_z2_t
                )

                # ---- Phase C contributions for x_2 ---------------------------
                # UP: pull_back ε_up_2 through W_up_2 → u_z2; gate by act'(x_2)
                UB2.pull_back_gpu[BATCH, dtype](ctx, u_eps2_t, up_p2_t, u_z2_t)
                UB2.act_derivative_mul_gpu[BATCH, dtype](
                    ctx, x2_t, u_z2_t, u_z2_t
                )
                # DOWN: pull_back ε_dn_1 through W_dn_1 → d_z1; gate by act'(x_2)
                DB1.pull_back_gpu[BATCH, dtype](ctx, d_eps1_t, dn_p1_t, d_z1_t)
                DB1.act_derivative_mul_gpu[BATCH, dtype](
                    ctx, x2_t, d_z1_t, d_z1_t
                )

                # ---- Phase D: apply combined dx to BOTH latents -------------
                # x_1: ε_self_up = ε_up_0 (its self ε); ε_self_dn = ε_dn_1
                ctx.enqueue_function[k_dx](
                    x1_t,
                    u_eps0_t,
                    u_z1_t,
                    d_eps1_t,
                    d_z2_t,
                    lr_x_s,
                    alpha_up_s,
                    alpha_down_s,
                    grid_dim=(dx_blocks,),
                    block_dim=(TPB,),
                )
                # x_2: ε_self_up = ε_up_1; ε_self_dn = ε_dn_0
                ctx.enqueue_function[k_dx](
                    x2_t,
                    u_eps1_t,
                    u_z2_t,
                    d_eps0_t,
                    d_z1_t,
                    lr_x_s,
                    alpha_up_s,
                    alpha_down_s,
                    grid_dim=(dx_blocks,),
                    block_dim=(TPB,),
                )

            # Weight grads (using post-inference ε)
            UB0.weight_grad_gpu[BATCH, dtype](ctx, u_eps0_t, u_a0_t, up_g0_t)
            UB1.weight_grad_gpu[BATCH, dtype](ctx, u_eps1_t, u_a1_t, up_g1_t)
            UB2.weight_grad_gpu[BATCH, dtype](ctx, u_eps2_t, u_a2_t, up_g2_t)
            DB0.weight_grad_gpu[BATCH, dtype](ctx, d_eps0_t, d_a0_t, dn_g0_t)
            DB1.weight_grad_gpu[BATCH, dtype](ctx, d_eps1_t, d_a1_t, dn_g1_t)
            DB2.weight_grad_gpu[BATCH, dtype](ctx, d_eps2_t, d_a2_t, dn_g2_t)

            # Scale DOWN grads by α_down
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

    # ── Eval: classification via UP forward_eval_gpu ─────────────────────────
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

    print(
        "  Per-pixel MSE between class-0 and class-9 generated images:",
        class_diff,
    )
    print("  Mean |gen pixel| across 10 classes:", gen_mean_mag)

    # Bogacz reports above 95% on this exact recipe.
    if test_acc >= 0.95:
        print(
            "\n  [PASS] full Bogacz notebook 5: classification ≥95% (got",
            test_acc,
            ")",
        )
    elif test_acc >= 0.85:
        print(
            "\n  [PARTIAL] classification ≥85% (got",
            test_acc,
            ") — close to Bogacz target",
        )
    elif test_acc >= 0.50:
        print("\n  [PARTIAL] classification ≥50% but < 85% (got", test_acc, ")")
    else:
        print("\n  [FAIL] classification too low:", test_acc)
        raise Error("bidirectional GPU full test failed")
    print("=== Done ===")
