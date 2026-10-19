"""Action-conditioned tPC (GPU) — Step 2 of PCN_WORLD_MODEL_ROADMAP.md.

GPU port of `test_action_conditioned_tpc.mojo`. Same architecture, same toy
env, same pass criterion (1-step prediction MSE < 0.01).

Per-batch rollouts are generated on host and uploaded as `[SEQ_LEN, BATCH]`
device buffers (column-major in time so each step's slice is contiguous).
Two helper kernels stitch the temporal loop on device:
  - `_set_action_target_kernel`: writes actions[t-1] into x_in[:, HIDDEN]
                                  and states[t] into y_tgt[:, 0]
  - `_set_prev_hidden_kernel`:    writes lat[:, 0:HIDDEN] into x_in[:, 0:HIDDEN]

Run:
    pixi run -e apple  mojo run -I . tests/pcn/test_action_conditioned_tpc_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/pcn/test_action_conditioned_tpc_gpu.mojo
"""

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from std.memory import alloc, memset
from std.random.philox import Random as PhiloxRandom
from std.time import perf_counter_ns
from noeira.core.tiles import Tile, tile

from noeira.nn.constants import DT as dtype
from noeira.experimental.pcn.pc_constants import TPB
from noeira.experimental.pcn.pc_initializer import PCXavier
from noeira.experimental.pcn.pc_optimizer import PCAdam
from noeira.experimental.pcn import (
    PCBlock,
    PCSequential,
    PCTanh,
    PCTrainer,
)


comptime BATCH = 32
comptime HIDDEN = 16
comptime ACTION_DIM = 1
comptime DATA_DIM = 1
comptime AUG_DIM = HIDDEN + ACTION_DIM
comptime SEQ_LEN = 10
comptime EPOCHS = 50
comptime N_BATCHES_PER_EPOCH = 50
comptime T_INFER = 50
comptime LR_X: Float64 = 0.01
comptime ADAM_LR: Float64 = 0.005

comptime ACTION_STEP: Float64 = 0.1

comptime NET = PCSequential[
    PCBlock[AUG_DIM, HIDDEN, PCTanh],
    PCBlock[HIDDEN, DATA_DIM, PCTanh],
]
comptime TRAINER = PCTrainer[
    PCBlock[AUG_DIM, HIDDEN, PCTanh],
    PCBlock[HIDDEN, DATA_DIM, PCTanh],
    dtype=dtype,
]
comptime OPT = PCAdam[LR=ADAM_LR]


def _set_action_target_kernel[
    BATCH: Int,
    AUG_DIM: Int,
    HIDDEN: Int,
    DATA_DIM: Int,
    KDT: DType,
](
    actions_slice: Tile[KDT, BATCH],
    states_slice: Tile[KDT, BATCH],
    x_in: Tile[KDT, BATCH, AUG_DIM],
    y_tgt: Tile[KDT, BATCH, DATA_DIM],
):
    """Write per-step action into x_in[:, HIDDEN] and target state into y_tgt[:, 0].
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return
    x_in[b, HIDDEN] = rebind[Scalar[KDT]](actions_slice[b])
    y_tgt[b, 0] = rebind[Scalar[KDT]](states_slice[b])


def _set_prev_hidden_kernel[
    BATCH: Int,
    AUG_DIM: Int,
    HIDDEN: Int,
    LATENT_DIM: Int,
    KDT: DType,
](
    latents: Tile[KDT, BATCH, LATENT_DIM],
    x_in: Tile[KDT, BATCH, AUG_DIM],
):
    """Copy lat[:, 0:HIDDEN] into x_in[:, 0:HIDDEN] (the recurrent prefix)."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * HIDDEN:
        return
    var b = idx // HIDDEN
    var k = idx % HIDDEN
    x_in[b, k] = rebind[Scalar[KDT]](latents[b, k])


def _zero_x_in_kernel[
    BATCH: Int,
    AUG_DIM: Int,
    KDT: DType,
](x_in: Tile[KDT, BATCH, AUG_DIM],):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * AUG_DIM:
        return
    var b = idx // AUG_DIM
    var k = idx % AUG_DIM
    x_in[b, k] = Scalar[KDT](0)


def _sample_action(mut rng: PhiloxRandom) -> Float64:
    var u = Float64(rng.step_uniform()[0])
    return -1.0 if u < 0.5 else 1.0


def main() raises:
    print("=" * 60)
    print("Action-conditioned tPC (GPU) — roadmap Step 2")
    print("=" * 60)
    print(
        "  arch       : PCBlock[",
        AUG_DIM,
        ",",
        HIDDEN,
        ",PCTanh] → PCBlock[",
        HIDDEN,
        ",",
        DATA_DIM,
        ",PCTanh]",
    )
    print("  PARAM_SIZE :", NET.PARAM_SIZE, "  LATENT_DIM:", NET.LATENT_DIM)
    print(
        "  BATCH=",
        BATCH,
        " SEQ_LEN=",
        SEQ_LEN,
        " EPOCHS=",
        EPOCHS,
        " N_BATCHES=",
        N_BATCHES_PER_EPOCH,
    )
    print("  T_INFER=", T_INFER, " LR_X=", LR_X, " ADAM_LR=", ADAM_LR)

    var ctx = DeviceContext()

    # ── Init params on host then upload to GPU ────────────────────────────────
    var params_host_init = ctx.enqueue_create_host_buffer[dtype](NET.PARAM_SIZE)
    for i in range(NET.PARAM_SIZE):
        params_host_init.unsafe_ptr()[i] = Scalar[dtype](0)
    var params_init_t = tile[NET.PARAM_SIZE](params_host_init.unsafe_ptr().as_unsafe_any_origin())
    NET.pc_init_params[PCXavier, dtype](params_init_t)

    var params_dbuf = ctx.enqueue_create_buffer[dtype](NET.PARAM_SIZE)
    ctx.enqueue_copy(params_dbuf, params_host_init)

    var grads_dbuf = ctx.enqueue_create_buffer[dtype](NET.PARAM_SIZE)
    var lat_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.LATENT_DIM)
    var mu_eps_dbuf = ctx.enqueue_create_buffer[dtype](
        BATCH * NET.SCRATCH_OUT_DIM
    )
    var a_below_dbuf = ctx.enqueue_create_buffer[dtype](
        BATCH * NET.SCRATCH_IN_DIM
    )
    var z_below_dbuf = ctx.enqueue_create_buffer[dtype](
        BATCH * NET.SCRATCH_IN_DIM
    )
    var dx_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.LATENT_DIM)
    var x_in_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * AUG_DIM)
    var y_tgt_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * DATA_DIM)

    var params_t = tile[NET.PARAM_SIZE](params_dbuf)
    var grads_t = tile[NET.PARAM_SIZE](grads_dbuf)
    var lat_t = tile[BATCH, NET.LATENT_DIM](lat_dbuf)
    var mu_eps_t = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_eps_dbuf)
    var a_below_t = tile[BATCH, NET.SCRATCH_IN_DIM](a_below_dbuf)
    var z_below_t = tile[BATCH, NET.SCRATCH_IN_DIM](z_below_dbuf)
    var dx_t = tile[BATCH, NET.LATENT_DIM](dx_dbuf)
    var x_in_t = tile[BATCH, AUG_DIM](x_in_dbuf)
    var y_tgt_t = tile[BATCH, DATA_DIM](y_tgt_dbuf)

    # Adam state on GPU
    var opt_state_dbuf = ctx.enqueue_create_buffer[dtype](
        NET.PARAM_SIZE * OPT.STATE_PER_PARAM
    )
    var opt_global_dbuf = ctx.enqueue_create_buffer[dtype](
        OPT.GLOBAL_STATE_SIZE
    )
    var opt_state_init = ctx.enqueue_create_host_buffer[dtype](
        NET.PARAM_SIZE * OPT.STATE_PER_PARAM
    )
    for i in range(NET.PARAM_SIZE * OPT.STATE_PER_PARAM):
        opt_state_init.unsafe_ptr()[i] = Scalar[dtype](0)
    ctx.enqueue_copy(opt_state_dbuf, opt_state_init)

    var opt_global_init = ctx.enqueue_create_host_buffer[dtype](
        OPT.GLOBAL_STATE_SIZE
    )
    opt_global_init.unsafe_ptr()[0] = Scalar[dtype](0)
    opt_global_init.unsafe_ptr()[1] = Scalar[dtype](1.0)
    ctx.enqueue_copy(opt_global_dbuf, opt_global_init)

    var opt_state_t = tile[NET.PARAM_SIZE, OPT.STATE_PER_PARAM](opt_state_dbuf)
    var opt_global_t = tile[OPT.GLOBAL_STATE_SIZE](opt_global_dbuf)

    # Per-batch rollout buffers — column-major in time so per-step slice contiguous.
    # actions[t, b] at offset t * BATCH + b; states[t, b] similarly.
    var actions_host = ctx.enqueue_create_host_buffer[dtype](SEQ_LEN * BATCH)
    var states_host = ctx.enqueue_create_host_buffer[dtype](
        (SEQ_LEN + 1) * BATCH
    )
    var actions_dbuf = ctx.enqueue_create_buffer[dtype](SEQ_LEN * BATCH)
    var states_dbuf = ctx.enqueue_create_buffer[dtype]((SEQ_LEN + 1) * BATCH)

    ctx.synchronize()

    # ── Train ────────────────────────────────────────────────────────────────
    print("\n  epoch | last_step_loss (skipped — no host sync) | wall_t (s)")
    print("  ------+---------------------------------------+------------")

    var rng = PhiloxRandom(seed=UInt64(7), offset=UInt64(0))
    var step_num: Int = 0
    var t0 = perf_counter_ns()

    for epoch in range(EPOCHS):
        for batch_idx in range(N_BATCHES_PER_EPOCH):
            # Generate rollouts on host: actions[t, b], states[t, b]
            for b in range(BATCH):
                states_host.unsafe_ptr()[0 * BATCH + b] = Scalar[dtype](0.0)
            for t in range(SEQ_LEN):
                for b in range(BATCH):
                    var a = _sample_action(rng)
                    actions_host.unsafe_ptr()[t * BATCH + b] = Scalar[dtype](a)
                    var s_prev = Float64(
                        states_host.unsafe_ptr()[t * BATCH + b]
                    )
                    states_host.unsafe_ptr()[(t + 1) * BATCH + b] = Scalar[
                        dtype
                    ](s_prev + ACTION_STEP * a)

            ctx.enqueue_copy(actions_dbuf, actions_host)
            ctx.enqueue_copy(states_dbuf, states_host)

            # Reset prev_hidden = 0 (fills HIDDEN prefix and action slot — both zeroed).
            comptime zero_k = _zero_x_in_kernel[BATCH, AUG_DIM, dtype]
            var zero_threads = BATCH * AUG_DIM
            var zero_blocks = (zero_threads + TPB - 1) // TPB
            ctx.enqueue_function[zero_k](
                x_in_t,
                grid_dim=(zero_blocks,),
                block_dim=(TPB,),
            )

            for t in range(1, SEQ_LEN + 1):
                var actions_slice = tile[BATCH](actions_dbuf.unsafe_ptr().as_unsafe_any_origin() + (t - 1) * BATCH)
                var states_slice = tile[BATCH](states_dbuf.unsafe_ptr().as_unsafe_any_origin() + t * BATCH)

                comptime sat_k = _set_action_target_kernel[
                    BATCH, AUG_DIM, HIDDEN, DATA_DIM, dtype
                ]
                var sat_blocks = (BATCH + TPB - 1) // TPB
                ctx.enqueue_function[sat_k](
                    actions_slice,
                    states_slice,
                    x_in_t,
                    y_tgt_t,
                    grid_dim=(sat_blocks,),
                    block_dim=(TPB,),
                )

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
                    y_tgt_t,
                    T_infer=T_INFER,
                    lr_x=Scalar[dtype](LR_X),
                )
                step_num += 1
                OPT.step_gpu[NET.PARAM_SIZE, dtype](
                    ctx, params_t, grads_t, opt_state_t, opt_global_t, step_num
                )

                # Copy lat[:, 0:HIDDEN] → x_in[:, 0:HIDDEN]
                comptime sph_k = _set_prev_hidden_kernel[
                    BATCH, AUG_DIM, HIDDEN, NET.LATENT_DIM, dtype
                ]
                var sph_threads = BATCH * HIDDEN
                var sph_blocks = (sph_threads + TPB - 1) // TPB
                ctx.enqueue_function[sph_k](
                    lat_t,
                    x_in_t,
                    grid_dim=(sph_blocks,),
                    block_dim=(TPB,),
                )

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            ctx.synchronize()
            var elapsed = Float64(perf_counter_ns() - t0) / 1e9
            print("    ", epoch, "  ", fit(String(elapsed), 7))

    ctx.synchronize()
    var total_t = Float64(perf_counter_ns() - t0) / 1e9
    print("\n  total train time:", total_t, "s")

    # ── Eval ──────────────────────────────────────────────────────────────────
    # Held-out rollouts. At each step:
    #   1. Predict s_t = block_1.predict(block_0.predict([prev_hidden, a_{t-1}]))
    #   2. Settle z_t against s_t (teacher forcing) → prev_hidden = z_t
    var eval_rng = PhiloxRandom(seed=UInt64(101), offset=UInt64(0))
    for b in range(BATCH):
        states_host.unsafe_ptr()[0 * BATCH + b] = Scalar[dtype](0.0)
    for t in range(SEQ_LEN):
        for b in range(BATCH):
            var a = _sample_action(eval_rng)
            actions_host.unsafe_ptr()[t * BATCH + b] = Scalar[dtype](a)
            var s_prev = Float64(states_host.unsafe_ptr()[t * BATCH + b])
            states_host.unsafe_ptr()[(t + 1) * BATCH + b] = Scalar[dtype](
                s_prev + ACTION_STEP * a
            )
    ctx.enqueue_copy(actions_dbuf, actions_host)
    ctx.enqueue_copy(states_dbuf, states_host)

    # Eval feedforward scratch
    var z_pred_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    var a_z_pred_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * AUG_DIM)
    var s_pred_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * DATA_DIM)
    var a_s_pred_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN)
    var z_pred_t = tile[BATCH, HIDDEN](z_pred_dbuf)
    var a_z_pred_t = tile[BATCH, AUG_DIM](a_z_pred_dbuf)
    var s_pred_t = tile[BATCH, DATA_DIM](s_pred_dbuf)
    var a_s_pred_t = tile[BATCH, HIDDEN](a_s_pred_dbuf)

    var s_pred_host = ctx.enqueue_create_host_buffer[dtype](BATCH * DATA_DIM)

    # Reset prev_hidden to zero
    comptime zero_k_eval = _zero_x_in_kernel[BATCH, AUG_DIM, dtype]
    var zero_threads_eval = BATCH * AUG_DIM
    var zero_blocks_eval = (zero_threads_eval + TPB - 1) // TPB
    ctx.enqueue_function[zero_k_eval](
        x_in_t,
        grid_dim=(zero_blocks_eval,),
        block_dim=(TPB,),
    )

    # Per-block param views (for eval feedforward)
    comptime offset_b1 = NET._param_offset[1]()
    var params_b0_t = tile[NET.block_types[0].PARAM_SIZE](params_dbuf.unsafe_ptr().as_unsafe_any_origin())
    var params_b1_t = tile[NET.block_types[1].PARAM_SIZE](params_dbuf.unsafe_ptr().as_unsafe_any_origin() + offset_b1)

    var total_sq_err: Float64 = 0.0
    var total_baseline_err: Float64 = 0.0
    var n_predictions: Int = 0

    print("\n  step | avg 1-step MSE | avg baseline MSE (predict-no-change)")
    print("  -----+----------------+-------------------------------------")

    for t in range(1, SEQ_LEN + 1):
        # x_in[:, HIDDEN] = action[t-1, :]
        var actions_slice = tile[BATCH](actions_dbuf.unsafe_ptr().as_unsafe_any_origin() + (t - 1) * BATCH)
        var states_slice = tile[BATCH](states_dbuf.unsafe_ptr().as_unsafe_any_origin() + t * BATCH)

        comptime sat_k_eval = _set_action_target_kernel[
            BATCH, AUG_DIM, HIDDEN, DATA_DIM, dtype
        ]
        var sat_blocks_eval = (BATCH + TPB - 1) // TPB
        ctx.enqueue_function[sat_k_eval](
            actions_slice,
            states_slice,
            x_in_t,
            y_tgt_t,
            grid_dim=(sat_blocks_eval,),
            block_dim=(TPB,),
        )

        # 1) Predict (no settle): z_pred = block_0(x_in); s_pred = block_1(z_pred)
        NET.block_types[0].predict_gpu[BATCH, dtype](
            ctx, x_in_t, params_b0_t, z_pred_t, a_z_pred_t
        )
        NET.block_types[1].predict_gpu[BATCH, dtype](
            ctx, z_pred_t, params_b1_t, s_pred_t, a_s_pred_t
from noeira.core.fmt import fit
        )

        # Download s_pred to host for MSE
        ctx.enqueue_copy(s_pred_host, s_pred_dbuf)
        ctx.synchronize()

        var step_mse: Float64 = 0.0
        var step_baseline: Float64 = 0.0
        for b in range(BATCH):
            var s_true = Float64(states_host.unsafe_ptr()[t * BATCH + b])
            var s_prev = Float64(states_host.unsafe_ptr()[(t - 1) * BATCH + b])
            var d = Float64(s_pred_host.unsafe_ptr()[b * DATA_DIM]) - s_true
            step_mse += d * d
            var d0 = s_prev - s_true
            step_baseline += d0 * d0
        step_mse /= Float64(BATCH)
        step_baseline /= Float64(BATCH)
        total_sq_err += step_mse
        total_baseline_err += step_baseline
        n_predictions += 1
        print("    ", t, "  ", step_mse, "    ", step_baseline)

        # 2) Settle z_t against s_t (teacher forcing)
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
            y_tgt_t,
            T_infer=T_INFER,
            lr_x=Scalar[dtype](LR_X),
        )
        # No Adam.step for eval

        comptime sph_k_eval = _set_prev_hidden_kernel[
            BATCH, AUG_DIM, HIDDEN, NET.LATENT_DIM, dtype
        ]
        var sph_threads_eval = BATCH * HIDDEN
        var sph_blocks_eval = (sph_threads_eval + TPB - 1) // TPB
        ctx.enqueue_function[sph_k_eval](
            lat_t,
            x_in_t,
            grid_dim=(sph_blocks_eval,),
            block_dim=(TPB,),
        )

    var avg_mse = total_sq_err / Float64(n_predictions)
    var avg_baseline = total_baseline_err / Float64(n_predictions)

    print("\n  avg 1-step MSE :", avg_mse)
    print("  avg baseline   :", avg_baseline)
    print(
        "  ratio          :",
        avg_mse / avg_baseline if avg_baseline > 0 else 1.0,
    )

    if avg_mse < 0.01:
        print(
            "\n  [PASS] action-conditioned tPC GPU: 1-step prediction MSE",
            avg_mse,
            "< 0.01",
        )
    else:
        print("\n  [FAIL] 1-step prediction MSE", avg_mse, "≥ 0.01")
        raise Error("action-conditioned tPC GPU test failed")

    print("=== Done ===")
