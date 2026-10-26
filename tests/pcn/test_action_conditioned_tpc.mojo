"""Action-conditioned tPC — Step 2 of PCN_WORLD_MODEL_ROADMAP.md.

Drives the recurrent PC dynamics with an exogenous control signal:
    z_t = W_r·tanh([z_{t-1}, a_{t-1}]) + b      (concat input)
    s_t = W_dec·tanh(z_t) + b_dec               (emission)

Default option (a) from the roadmap: concat `[z_{t-1}, a_{t-1}]` into the
recurrent block's input — zero framework change. The block's W naturally
splits into [W_r | W_in] structurally; we don't expose that decomposition.

Toy environment (1D bang-bang): s_0 = 0, a_t ∈ {-1, +1} uniform, deterministic
transition s_{t+1} = s_t + 0.1·a_t. With SEQ_LEN=10 we keep |s_T| ≤ 1.0 so
the tanh-input recurrence stays informative.

Training: BATCH random rollouts in parallel, temporal tPC over SEQ_LEN steps.
Eval: held-out rollouts, walk forward step-by-step. At each step predict s_t
from (prev_hidden, a_{t-1}) via feedforward (no settle), compute MSE vs
ground truth, then settle z_t against s_t (teacher-forced) before continuing.

Pass criterion: avg 1-step prediction MSE < 0.01 (the "predict no change"
baseline of E[(0.1·a)²] = 0.01).

Run:
    pixi run mojo run -I . tests/pcn/test_action_conditioned_tpc.mojo
"""

from std.memory import alloc, unsafe_memset
from std.random.philox import Random as PhiloxRandom
from std.time import perf_counter_ns
from noeira.core.tiles import tile

from noeira.nn.constants import DT as dtype
from noeira.core.fmt import fit
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
comptime AUG_DIM = HIDDEN + ACTION_DIM        # block_0 input width
comptime SEQ_LEN = 10
comptime EPOCHS = 50
comptime N_BATCHES_PER_EPOCH = 50
comptime T_INFER = 50
comptime LR_X: Float64 = 0.01
comptime ADAM_LR: Float64 = 0.005

comptime ACTION_STEP: Float64 = 0.1           # transition magnitude

comptime NET = PCSequential[
    PCBlock[AUG_DIM, HIDDEN, PCTanh],         # action-conditioned recurrence
    PCBlock[HIDDEN, DATA_DIM, PCTanh],        # emission
]
comptime TRAINER = PCTrainer[
    PCBlock[AUG_DIM, HIDDEN, PCTanh],
    PCBlock[HIDDEN, DATA_DIM, PCTanh],
    dtype=dtype,
]
comptime OPT = PCAdam[LR=ADAM_LR]


def _sample_action(mut rng: PhiloxRandom) -> Float64:
    """Uniform {-1, +1}."""
    var u = Float64(rng.step_uniform()[0])
    return -1.0 if u < 0.5 else 1.0


def main() raises:
    print("=" * 60)
    print("Action-conditioned tPC — roadmap Step 2")
    print("=" * 60)
    print("  arch       : PCBlock[", AUG_DIM, ",", HIDDEN, ",PCTanh] → PCBlock[", HIDDEN, ",", DATA_DIM, ",PCTanh]")
    print("  PARAM_SIZE :", NET.PARAM_SIZE, "  LATENT_DIM:", NET.LATENT_DIM)
    print("  BATCH=", BATCH, " SEQ_LEN=", SEQ_LEN, " EPOCHS=", EPOCHS, " N_BATCHES=", N_BATCHES_PER_EPOCH)
    print("  T_INFER=", T_INFER, " LR_X=", LR_X, " ADAM_LR=", ADAM_LR)
    print("  env        : 1D bang-bang, s_{t+1}=s_t+", ACTION_STEP, "·a_t,  a∈{-1,+1}")

    # ── Allocate net params + Adam state ──────────────────────────────────────
    var params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var grads_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var opt_state_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE * OPT.STATE_PER_PARAM).as_unsafe_any_origin()
    var opt_global_buf = alloc[Scalar[dtype]](OPT.GLOBAL_STATE_SIZE).as_unsafe_any_origin()
    unsafe_memset(params_buf, 0, NET.PARAM_SIZE)
    unsafe_memset(grads_buf, 0, NET.PARAM_SIZE)
    unsafe_memset(opt_state_buf, 0, NET.PARAM_SIZE * OPT.STATE_PER_PARAM)
    unsafe_memset(opt_global_buf, 0, OPT.GLOBAL_STATE_SIZE)

    var params = tile[NET.PARAM_SIZE](params_buf)
    var grads = tile[NET.PARAM_SIZE](grads_buf)
    var opt_state = tile[NET.PARAM_SIZE, OPT.STATE_PER_PARAM](opt_state_buf)
    var opt_global = tile[OPT.GLOBAL_STATE_SIZE](opt_global_buf)
    NET.pc_init_params[PCXavier, dtype](params)

    # ── Scratch buffers ───────────────────────────────────────────────────────
    var lat_buf = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var mu_eps_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_OUT_DIM).as_unsafe_any_origin()
    var a_below_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var z_below_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var dx_buf_raw = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    unsafe_memset(lat_buf, 0, BATCH * NET.LATENT_DIM)
    unsafe_memset(mu_eps_buf_raw, 0, BATCH * NET.SCRATCH_OUT_DIM)
    unsafe_memset(a_below_buf_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
    unsafe_memset(z_below_buf_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
    unsafe_memset(dx_buf_raw, 0, BATCH * NET.LATENT_DIM)

    var latents = tile[BATCH, NET.LATENT_DIM](lat_buf)
    var mu_eps_buf = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_eps_buf_raw)
    var a_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](a_below_buf_raw)
    var z_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](z_below_buf_raw)
    var dx_buf = tile[BATCH, NET.LATENT_DIM](dx_buf_raw)

    # ── Per-step input + target buffers ───────────────────────────────────────
    var x_in_buf = alloc[Scalar[dtype]](BATCH * AUG_DIM).as_unsafe_any_origin()
    var y_tgt_buf = alloc[Scalar[dtype]](BATCH * DATA_DIM).as_unsafe_any_origin()
    unsafe_memset(x_in_buf, 0, BATCH * AUG_DIM)
    unsafe_memset(y_tgt_buf, 0, BATCH * DATA_DIM)
    var x_in = tile[BATCH, AUG_DIM](x_in_buf)
    var y_target = tile[BATCH, DATA_DIM](y_tgt_buf)

    # Per-rollout scratch — actions [BATCH, SEQ_LEN] and states [BATCH, SEQ_LEN+1].
    var actions_buf = alloc[Scalar[dtype]](BATCH * SEQ_LEN).as_unsafe_any_origin()
    var states_buf = alloc[Scalar[dtype]](BATCH * (SEQ_LEN + 1)).as_unsafe_any_origin()

    # Param views per block (for eval feedforward)
    comptime offset_b1 = NET._param_offset[1]()
    var params_b0 = tile[NET.block_types[0].PARAM_SIZE](params_buf)
    var params_b1 = tile[NET.block_types[1].PARAM_SIZE](params_buf + offset_b1)

    # ── Train ────────────────────────────────────────────────────────────────
    print("\n  epoch | last_step_loss | wall_t (s)")
    print("  ------+----------------+------------")

    var step_num: Int = 0
    var rng = PhiloxRandom(seed=UInt64(7), offset=UInt64(0))
    var t0 = perf_counter_ns()

    for epoch in range(EPOCHS):
        var last_loss: Float64 = 0.0
        for batch_idx in range(N_BATCHES_PER_EPOCH):
            # Generate BATCH rollouts (random action sequences).
            for b in range(BATCH):
                states_buf[b * (SEQ_LEN + 1) + 0] = Scalar[dtype](0.0)
                for t in range(SEQ_LEN):
                    var a = _sample_action(rng)
                    actions_buf[b * SEQ_LEN + t] = Scalar[dtype](a)
                    var s_prev = Float64(states_buf[b * (SEQ_LEN + 1) + t])
                    states_buf[b * (SEQ_LEN + 1) + t + 1] = Scalar[dtype](
                        s_prev + ACTION_STEP * a
                    )

            # Reset prev_hidden to zero (s_0 = 0 ⇒ z_0 ≡ 0).
            unsafe_memset(x_in_buf, 0, BATCH * AUG_DIM)

            for t in range(1, SEQ_LEN + 1):
                # x_in[b] = [prev_hidden[b], a[b, t-1]] — prev_hidden already
                # in x_in_buf[b, :HIDDEN] from the previous step (or zero at t=1).
                for b in range(BATCH):
                    x_in_buf[b * AUG_DIM + HIDDEN] = actions_buf[
                        b * SEQ_LEN + (t - 1)
                    ]
                    y_tgt_buf[b * DATA_DIM] = states_buf[b * (SEQ_LEN + 1) + t]

                var result = TRAINER.compute_grads_only[BATCH](
                    params, grads, latents,
                    mu_eps_buf, a_below_buf, z_below_buf, dx_buf,
                    x_in, y_target,
                    T_infer=T_INFER,
                    lr_x=Scalar[dtype](LR_X),
                )
                step_num += 1
                OPT.step[NET.PARAM_SIZE, dtype](
                    params, grads, opt_state, opt_global, step_num
                )
                last_loss = result.output_loss_final

                # prev_hidden = z_t = lat[:, 0:HIDDEN].
                # NOTE: x_in_buf[b, HIDDEN] (the action slot) gets overwritten
                # at the top of the next iteration, so we just refill the
                # hidden prefix here.
                for b in range(BATCH):
                    for j in range(HIDDEN):
                        x_in_buf[b * AUG_DIM + j] = lat_buf[b * NET.LATENT_DIM + j]

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            var elapsed = Float64(perf_counter_ns() - t0) / 1e9
            print(
                "    ", epoch, "  ",
                fit(String(last_loss), 11), "  ",
                fit(String(elapsed), 7),
            )

    var total_t = Float64(perf_counter_ns() - t0) / 1e9
    print("\n  total train time:", total_t, "s")

    # ── Eval ──────────────────────────────────────────────────────────────────
    # Held-out rollouts (different RNG seed). For each step:
    #   1. Predict s_t = decode(transition([prev_hidden, a_{t-1}]))   (no settle)
    #   2. Settle z_t against s_t (teacher-forced) → prev_hidden = z_t
    var eval_rng = PhiloxRandom(seed=UInt64(101), offset=UInt64(0))

    # Generate eval rollouts.
    for b in range(BATCH):
        states_buf[b * (SEQ_LEN + 1) + 0] = Scalar[dtype](0.0)
        for t in range(SEQ_LEN):
            var a = _sample_action(eval_rng)
            actions_buf[b * SEQ_LEN + t] = Scalar[dtype](a)
            var s_prev = Float64(states_buf[b * (SEQ_LEN + 1) + t])
            states_buf[b * (SEQ_LEN + 1) + t + 1] = Scalar[dtype](
                s_prev + ACTION_STEP * a
            )

    # prev_hidden ← 0
    unsafe_memset(x_in_buf, 0, BATCH * AUG_DIM)

    # Feedforward scratch for the eval predict path.
    var z_pred_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var a_z_pred_buf = alloc[Scalar[dtype]](BATCH * AUG_DIM).as_unsafe_any_origin()
    var s_pred_buf = alloc[Scalar[dtype]](BATCH * DATA_DIM).as_unsafe_any_origin()
    var a_s_pred_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var z_pred = tile[BATCH, HIDDEN](z_pred_buf)
    var a_z_pred = tile[BATCH, AUG_DIM](a_z_pred_buf)
    var s_pred = tile[BATCH, DATA_DIM](s_pred_buf)
    var a_s_pred = tile[BATCH, HIDDEN](a_s_pred_buf)

    var total_sq_err: Float64 = 0.0
    var total_baseline_err: Float64 = 0.0
    var n_predictions: Int = 0

    print("\n  step | avg 1-step MSE | avg baseline MSE (predict-no-change)")
    print("  -----+----------------+-------------------------------------")

    for t in range(1, SEQ_LEN + 1):
        # Build x_in = [prev_hidden, a_{t-1}].
        for b in range(BATCH):
            x_in_buf[b * AUG_DIM + HIDDEN] = actions_buf[b * SEQ_LEN + (t - 1)]

        # 1) Predict without settle: z_pred = block_0(x_in); s_pred = block_1(z_pred).
        NET.block_types[0].predict[BATCH, dtype](x_in, params_b0, z_pred, a_z_pred)
        NET.block_types[1].predict[BATCH, dtype](z_pred, params_b1, s_pred, a_s_pred)

        # 2) Accumulate MSE vs ground truth.
        var step_mse: Float64 = 0.0
        var step_baseline: Float64 = 0.0
        for b in range(BATCH):
            var s_true = Float64(states_buf[b * (SEQ_LEN + 1) + t])
            var s_prev = Float64(states_buf[b * (SEQ_LEN + 1) + (t - 1)])
            var d = Float64(s_pred_buf[b * DATA_DIM]) - s_true
            step_mse += d * d
            var d0 = s_prev - s_true
            step_baseline += d0 * d0
        step_mse /= Float64(BATCH)
        step_baseline /= Float64(BATCH)
        total_sq_err += step_mse
        total_baseline_err += step_baseline
        n_predictions += 1
        print("    ", t, "  ", step_mse, "    ", step_baseline)

        # 3) Settle z_t against s_t for teacher forcing.
        for b in range(BATCH):
            y_tgt_buf[b * DATA_DIM] = states_buf[b * (SEQ_LEN + 1) + t]
        _ = TRAINER.compute_grads_only[BATCH](
            params, grads, latents,
            mu_eps_buf, a_below_buf, z_below_buf, dx_buf,
            x_in, y_target,
            T_infer=T_INFER,
            lr_x=Scalar[dtype](LR_X),
        )
        # No Adam.step — params frozen during eval.
        for b in range(BATCH):
            for j in range(HIDDEN):
                x_in_buf[b * AUG_DIM + j] = lat_buf[b * NET.LATENT_DIM + j]

    var avg_mse = total_sq_err / Float64(n_predictions)
    var avg_baseline = total_baseline_err / Float64(n_predictions)

    print("\n  avg 1-step MSE :", avg_mse)
    print("  avg baseline   :", avg_baseline)
    print("  ratio          :", avg_mse / avg_baseline if avg_baseline > 0 else 1.0)

    if avg_mse < 0.01:
        print("\n  [PASS] action-conditioned tPC: 1-step prediction MSE", avg_mse, "< 0.01")
    else:
        print("\n  [FAIL] 1-step prediction MSE", avg_mse, "≥ 0.01")
        raise Error("action-conditioned tPC test failed")

    # cleanup
    params_buf.free()
    grads_buf.free()
    opt_state_buf.free()
    opt_global_buf.free()
    lat_buf.free()
    mu_eps_buf_raw.free()
    a_below_buf_raw.free()
    z_below_buf_raw.free()
    dx_buf_raw.free()
    x_in_buf.free()
    y_tgt_buf.free()
    actions_buf.free()
    states_buf.free()
    z_pred_buf.free()
    a_z_pred_buf.free()
    s_pred_buf.free()
    a_s_pred_buf.free()
    print("=== Done ===")
