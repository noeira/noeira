"""World-model dynamics — Step 3 of PCN_WORLD_MODEL_ROADMAP.md.

Combines Steps 1 + 2: stochastic action-conditioned tPC. The latent transition
model is p(z_t | z_{t-1}, a_{t-1}) implemented with concat input + SGLD
posterior:
    z_t  ~  W_r·tanh([z_{t-1}, a_{t-1}]) + b   +   SGLD noise on inference
    s_t  =  W_dec·tanh(z_t) + b_dec            (emission)

This is the latent transition model used by Dreamer/PlaNet/TD-MPC2 (their
amortized encoders just replace the iterative SGLD chain).

Toy environment: 1D point with stochastic bang-bang dynamics
    s_0 = 0,  a_t ∈ {-1, +1},  s_{t+1} = s_t + 0.1·a_t + N(0, σ_env²)

Training: BATCH random rollouts with stochastic transitions, compute_grads_only_mcpc.

Eval: imagined rollouts. Fixed action sequence (all +1) so ground-truth
mean grows monotonically (s_t_mean = 0.1·t, s_t_var = t·σ_env²). N=100 imagined
trajectories generated in parallel via generate_samples per time step. Compare
empirical (mean, var) per step to N_MC=10000 Monte Carlo env rollouts.

Pass criterion (averaged across steps t=1..SEQ_LEN):
- Mean relative error ≤ 10%
- Variance relative error ≤ 30%

Run:
    pixi run mojo run -I . tests/pcn/test_world_model_dynamics.mojo
"""

from std.math import sqrt, log, cos, pi
from std.memory import alloc, unsafe_memset
from std.random.philox import Random as PhiloxRandom
from std.time import perf_counter_ns
from noeira.core.tiles import tile

from noeira.nn.constants import DT as dtype
from noeira.experimental.pcn.pc_initializer import PCXavier
from noeira.experimental.pcn.pc_optimizer import PCAdam
from noeira.experimental.pcn import (
    PCBlock,
    PCSequential,
    PCTanh,
    PCTrainer,
)


comptime BATCH = 32                       # training rollouts in parallel
comptime EVAL_BATCH = 500                 # imagined rollouts per step
comptime HIDDEN = 32
comptime ACTION_DIM = 1
comptime DATA_DIM = 1
comptime AUG_DIM = HIDDEN + ACTION_DIM
comptime SEQ_LEN = 5
comptime EPOCHS = 160
comptime N_BATCHES_PER_EPOCH = 50
comptime T_MIXING = 50
comptime T_SAMPLING = 1
comptime T_GENERATE = 500                 # SGLD steps per imagined-rollout step
comptime LR_X: Float64 = 0.01
comptime ADAM_LR: Float64 = 0.003
comptime ACTION_STEP: Float64 = 0.1
comptime SIGMA_ENV: Float64 = 0.05        # env transition noise std
comptime SGLD_NOISE_VAR: Float64 = 0.0008  # tuned for ||W_dec||² scaling

# Ground-truth Monte Carlo
comptime N_MC = 10000

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

# Bump per SGLD step to keep substreams disjoint (Box-Muller uses 2 uniforms).
comptime PHILOX_BUMP_PER_TRAIN_STEP = UInt64(
    BATCH * NET.LATENT_DIM * (T_MIXING + T_SAMPLING) * 2
)
comptime PHILOX_BUMP_PER_GEN_STEP = UInt64(
    EVAL_BATCH * NET.LATENT_DIM * T_GENERATE * 2
)


def _gauss_n01(mut rng: PhiloxRandom) -> Float64:
    """Box-Muller, returns one N(0,1) sample."""
    var u1 = Float64(rng.step_uniform()[0])
    var u2 = Float64(rng.step_uniform()[0])
    if u1 < 1e-10:
        u1 = 1e-10
    return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)


def _sample_action(mut rng: PhiloxRandom) -> Float64:
    """Uniform {-1, +1}."""
    var u = Float64(rng.step_uniform()[0])
    return -1.0 if u < 0.5 else 1.0


def main() raises:
    print("=" * 60)
    print("World-model dynamics — roadmap Step 3")
    print("=" * 60)
    print("  arch       : PCBlock[", AUG_DIM, ",", HIDDEN, ",PCTanh] → PCBlock[", HIDDEN, ",", DATA_DIM, ",PCTanh]")
    print("  PARAM_SIZE :", NET.PARAM_SIZE, "  LATENT_DIM:", NET.LATENT_DIM)
    print("  BATCH(train)=", BATCH, "  EVAL_BATCH=", EVAL_BATCH, "  SEQ_LEN=", SEQ_LEN)
    print("  EPOCHS=", EPOCHS, "  N_BATCHES=", N_BATCHES_PER_EPOCH, "  T_MIXING=", T_MIXING, "  T_GENERATE=", T_GENERATE)
    print("  env σ_env  :", SIGMA_ENV, " (transition noise std)")
    print("  SGLD       : lr_x=", LR_X, " noise_var=", SGLD_NOISE_VAR)

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

    # ── Training scratch (BATCH=32) ───────────────────────────────────────────
    var lat_buf = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var mu_eps_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_OUT_DIM).as_unsafe_any_origin()
    var a_below_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var z_below_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var dx_buf_raw = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var noise_buf_raw = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    unsafe_memset(lat_buf, 0, BATCH * NET.LATENT_DIM)
    unsafe_memset(mu_eps_buf_raw, 0, BATCH * NET.SCRATCH_OUT_DIM)
    unsafe_memset(a_below_buf_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
    unsafe_memset(z_below_buf_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
    unsafe_memset(dx_buf_raw, 0, BATCH * NET.LATENT_DIM)
    unsafe_memset(noise_buf_raw, 0, BATCH * NET.LATENT_DIM)

    var latents = tile[BATCH, NET.LATENT_DIM](lat_buf)
    var mu_eps_buf = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_eps_buf_raw)
    var a_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](a_below_buf_raw)
    var z_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](z_below_buf_raw)
    var dx_buf = tile[BATCH, NET.LATENT_DIM](dx_buf_raw)
    var noise_buf = tile[BATCH, NET.LATENT_DIM](noise_buf_raw)

    # Per-step input + target buffers (training).
    var x_in_buf = alloc[Scalar[dtype]](BATCH * AUG_DIM).as_unsafe_any_origin()
    var y_tgt_buf = alloc[Scalar[dtype]](BATCH * DATA_DIM).as_unsafe_any_origin()
    unsafe_memset(x_in_buf, 0, BATCH * AUG_DIM)
    unsafe_memset(y_tgt_buf, 0, BATCH * DATA_DIM)
    var x_in = tile[BATCH, AUG_DIM](x_in_buf)
    var y_target = tile[BATCH, DATA_DIM](y_tgt_buf)

    # Per-rollout actions/states scratch (training).
    var actions_buf = alloc[Scalar[dtype]](BATCH * SEQ_LEN).as_unsafe_any_origin()
    var states_buf = alloc[Scalar[dtype]](BATCH * (SEQ_LEN + 1)).as_unsafe_any_origin()

    # ── Train ────────────────────────────────────────────────────────────────
    print("\n  epoch | last_step_loss | wall_t (s)")
    print("  ------+----------------+------------")

    var step_num: Int = 0
    var rng = PhiloxRandom(seed=UInt64(7), offset=UInt64(0))
    var noise_offset: UInt64 = 1_000_000
    var philox_seed: UInt64 = 42
    var t0 = perf_counter_ns()

    for epoch in range(EPOCHS):
        var last_loss: Float64 = 0.0
        for batch_idx in range(N_BATCHES_PER_EPOCH):
            # Generate stochastic rollouts.
            for b in range(BATCH):
                states_buf[b * (SEQ_LEN + 1) + 0] = Scalar[dtype](0.0)
                for t in range(SEQ_LEN):
                    var a = _sample_action(rng)
                    actions_buf[b * SEQ_LEN + t] = Scalar[dtype](a)
                    var noise = SIGMA_ENV * _gauss_n01(rng)
                    var s_prev = Float64(states_buf[b * (SEQ_LEN + 1) + t])
                    states_buf[b * (SEQ_LEN + 1) + t + 1] = Scalar[dtype](
                        s_prev + ACTION_STEP * a + noise
                    )

            # Reset prev_hidden = 0 for each rollout (s_0 = 0 ⇒ z_0 ≡ 0).
            unsafe_memset(x_in_buf, 0, BATCH * AUG_DIM)

            for t in range(1, SEQ_LEN + 1):
                for b in range(BATCH):
                    x_in_buf[b * AUG_DIM + HIDDEN] = actions_buf[
                        b * SEQ_LEN + (t - 1)
                    ]
                    y_tgt_buf[b * DATA_DIM] = states_buf[b * (SEQ_LEN + 1) + t]

                var result = TRAINER.compute_grads_only_mcpc[BATCH](
                    params, grads, latents,
                    mu_eps_buf, a_below_buf, z_below_buf, dx_buf, noise_buf,
                    x_in, y_target,
                    T_mixing=T_MIXING,
                    T_sampling=T_SAMPLING,
                    lr_x=Scalar[dtype](LR_X),
                    noise_var=Scalar[dtype](SGLD_NOISE_VAR),
                    seed=philox_seed,
                    offset_base=noise_offset,
                )
                noise_offset += PHILOX_BUMP_PER_TRAIN_STEP

                step_num += 1
                OPT.step[NET.PARAM_SIZE, dtype](
                    params, grads, opt_state, opt_global, step_num
                )
                last_loss = result.output_loss_final

                # prev_hidden = z_t = lat[:, 0:HIDDEN].
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

    # Free training scratch — eval allocates EVAL_BATCH-sized buffers.
    lat_buf.free()
    mu_eps_buf_raw.free()
    a_below_buf_raw.free()
    z_below_buf_raw.free()
    dx_buf_raw.free()
    noise_buf_raw.free()
    x_in_buf.free()
    y_tgt_buf.free()
    actions_buf.free()
    states_buf.free()

    # ── Eval: fixed action sequence (all +1) ──────────────────────────────────
    # Ground-truth analytical: s_t_mean = 0.1·t, s_t_var = t·σ_env².
    # Ground-truth Monte Carlo: simulate N_MC env rollouts with same actions.
    print("\n  [eval] Monte Carlo ground truth: N_MC=", N_MC)
    var mc_mean = alloc[Float64](SEQ_LEN + 1).as_unsafe_any_origin()
    var mc_var = alloc[Float64](SEQ_LEN + 1).as_unsafe_any_origin()
    for t in range(SEQ_LEN + 1):
        mc_mean[t] = 0.0
        mc_var[t] = 0.0

    var mc_rng = PhiloxRandom(seed=UInt64(2026), offset=UInt64(0))
    var mc_states = alloc[Float64](N_MC * (SEQ_LEN + 1)).as_unsafe_any_origin()
    for n in range(N_MC):
        mc_states[n * (SEQ_LEN + 1) + 0] = 0.0
        for t in range(SEQ_LEN):
            var noise = SIGMA_ENV * _gauss_n01(mc_rng)
            var s_prev = mc_states[n * (SEQ_LEN + 1) + t]
            mc_states[n * (SEQ_LEN + 1) + t + 1] = (
                s_prev + ACTION_STEP * 1.0 + noise
            )
    for t in range(SEQ_LEN + 1):
        var sum_v: Float64 = 0
        for n in range(N_MC):
            sum_v += mc_states[n * (SEQ_LEN + 1) + t]
        mc_mean[t] = sum_v / Float64(N_MC)
        var sum_sq: Float64 = 0
        for n in range(N_MC):
            var d = mc_states[n * (SEQ_LEN + 1) + t] - mc_mean[t]
            sum_sq += d * d
        mc_var[t] = sum_sq / Float64(N_MC)
    mc_states.free()

    # ── Imagined rollouts ────────────────────────────────────────────────────
    print("\n  [eval] generating", EVAL_BATCH, "imagined rollouts via generate_samples per step")

    var eval_lat_buf = alloc[Scalar[dtype]](EVAL_BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var eval_mu_eps_raw = alloc[Scalar[dtype]](EVAL_BATCH * NET.SCRATCH_OUT_DIM).as_unsafe_any_origin()
    var eval_a_below_raw = alloc[Scalar[dtype]](EVAL_BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var eval_z_below_raw = alloc[Scalar[dtype]](EVAL_BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var eval_dx_raw = alloc[Scalar[dtype]](EVAL_BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var eval_noise_raw = alloc[Scalar[dtype]](EVAL_BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var eval_x_in_buf = alloc[Scalar[dtype]](EVAL_BATCH * AUG_DIM).as_unsafe_any_origin()
    var eval_y_dummy_buf = alloc[Scalar[dtype]](EVAL_BATCH * DATA_DIM).as_unsafe_any_origin()
    var eval_sample_buf = alloc[Scalar[dtype]](EVAL_BATCH * DATA_DIM).as_unsafe_any_origin()
    unsafe_memset(eval_lat_buf, 0, EVAL_BATCH * NET.LATENT_DIM)
    unsafe_memset(eval_mu_eps_raw, 0, EVAL_BATCH * NET.SCRATCH_OUT_DIM)
    unsafe_memset(eval_a_below_raw, 0, EVAL_BATCH * NET.SCRATCH_IN_DIM)
    unsafe_memset(eval_z_below_raw, 0, EVAL_BATCH * NET.SCRATCH_IN_DIM)
    unsafe_memset(eval_dx_raw, 0, EVAL_BATCH * NET.LATENT_DIM)
    unsafe_memset(eval_noise_raw, 0, EVAL_BATCH * NET.LATENT_DIM)
    unsafe_memset(eval_x_in_buf, 0, EVAL_BATCH * AUG_DIM)
    unsafe_memset(eval_y_dummy_buf, 0, EVAL_BATCH * DATA_DIM)
    unsafe_memset(eval_sample_buf, 0, EVAL_BATCH * DATA_DIM)

    var eval_lat = tile[EVAL_BATCH, NET.LATENT_DIM](eval_lat_buf)
    var eval_mu_eps = tile[EVAL_BATCH, NET.SCRATCH_OUT_DIM](eval_mu_eps_raw)
    var eval_a_below = tile[EVAL_BATCH, NET.SCRATCH_IN_DIM](eval_a_below_raw)
    var eval_z_below = tile[EVAL_BATCH, NET.SCRATCH_IN_DIM](eval_z_below_raw)
    var eval_dx = tile[EVAL_BATCH, NET.LATENT_DIM](eval_dx_raw)
    var eval_noise = tile[EVAL_BATCH, NET.LATENT_DIM](eval_noise_raw)
    var eval_x_in = tile[EVAL_BATCH, AUG_DIM](eval_x_in_buf)
    var eval_y_dummy = tile[EVAL_BATCH, DATA_DIM](eval_y_dummy_buf)
    var eval_sample = tile[EVAL_BATCH, DATA_DIM](eval_sample_buf)

    # Imagined trajectories: imagined[n, t] for t = 1..SEQ_LEN.
    var imagined = alloc[Float64](EVAL_BATCH * SEQ_LEN).as_unsafe_any_origin()

    var eval_offset: UInt64 = 5_000_000_000
    var eval_seed: UInt64 = 99

    for t in range(1, SEQ_LEN + 1):
        # x_in[n] = [prev_hidden[n], +1.0].
        for n in range(EVAL_BATCH):
            eval_x_in_buf[n * AUG_DIM + HIDDEN] = Scalar[dtype](1.0)

        TRAINER.generate_samples[EVAL_BATCH](
            params, eval_lat, eval_mu_eps, eval_a_below, eval_z_below,
            eval_dx, eval_noise,
            eval_x_in, eval_y_dummy, eval_sample,
            T=T_GENERATE,
            lr_x=Scalar[dtype](LR_X),
            noise_var=Scalar[dtype](SGLD_NOISE_VAR),
            seed=eval_seed,
            offset_base=eval_offset,
        )
        eval_offset += PHILOX_BUMP_PER_GEN_STEP

        for n in range(EVAL_BATCH):
            imagined[n * SEQ_LEN + (t - 1)] = Float64(
                eval_sample_buf[n * DATA_DIM]
            )
            # prev_hidden[n] = z_t (post-SGLD latent).
            for j in range(HIDDEN):
                eval_x_in_buf[n * AUG_DIM + j] = eval_lat_buf[
                    n * NET.LATENT_DIM + j
                ]

    # ── Compare imagined stats to MC ground truth ────────────────────────────
    print("\n  step | imagined mean | true mean | rel-err |  imag var | true var | rel-err")
    print("  -----+---------------+-----------+---------+-----------+----------+---------")

    var mean_rel_err_total: Float64 = 0.0
    var var_rel_err_total: Float64 = 0.0
    for t in range(1, SEQ_LEN + 1):
        var sum_v: Float64 = 0
        for n in range(EVAL_BATCH):
            sum_v += imagined[n * SEQ_LEN + (t - 1)]
        var imag_mean = sum_v / Float64(EVAL_BATCH)

        var sum_sq: Float64 = 0
        for n in range(EVAL_BATCH):
            var d = imagined[n * SEQ_LEN + (t - 1)] - imag_mean
            sum_sq += d * d
        var imag_var = sum_sq / Float64(EVAL_BATCH)

        var true_mean = mc_mean[t]
        var true_var = mc_var[t]

        var mean_err = imag_mean - true_mean
        if mean_err < 0:
            mean_err = -mean_err
        var mean_rel = mean_err / true_mean if true_mean != 0.0 else mean_err

        var var_err = imag_var - true_var
        if var_err < 0:
            var_err = -var_err
        var var_rel = var_err / true_var if true_var != 0.0 else var_err

        mean_rel_err_total += mean_rel
        var_rel_err_total += var_rel

        print(
            "    ", t,
            "    ", fit(String(imag_mean), 9),
            "    ", fit(String(true_mean), 7),
            "  ", fit(String(mean_rel), 7),
            "    ", fit(String(imag_var), 9),
            "  ", fit(String(true_var), 7),
            "  ", fit(String(var_rel), 7),
        )

    var avg_mean_rel = mean_rel_err_total / Float64(SEQ_LEN)
    var avg_var_rel = var_rel_err_total / Float64(SEQ_LEN)

    print("\n  avg mean rel err :", avg_mean_rel)
    print("  avg var  rel err :", avg_var_rel)

    # ── Predictive log-likelihood / Gaussian KL ──────────────────────────────
    # KL(env_t || model_t) under the Gaussian assumption; lower = better, 0 = oracle.
    # Persistence baseline marginalizes over the random action: predict s_t ~
    # N(0.1·(t-1), σ²_action + (t-1)·σ²_env) where σ²_action = ACTION_STEP² is
    # the variance contributed by an unknown ±1 action.
    print("\n  step | KL(env || model) | KL(env || persistence) | gain (persist−model)")
    print("  -----+------------------+------------------------+--------------------")
    var kl_model_total: Float64 = 0.0
    var kl_persist_total: Float64 = 0.0
    var var_action: Float64 = ACTION_STEP * ACTION_STEP
    var var_env_step: Float64 = SIGMA_ENV * SIGMA_ENV

    for t in range(1, SEQ_LEN + 1):
        var sum_v: Float64 = 0
        for n in range(EVAL_BATCH):
            sum_v += imagined[n * SEQ_LEN + (t - 1)]
        var imag_mean = sum_v / Float64(EVAL_BATCH)
        var sum_sq: Float64 = 0
        for n in range(EVAL_BATCH):
            var d = imagined[n * SEQ_LEN + (t - 1)] - imag_mean
            sum_sq += d * d
        # Sample-variance with floor (avoid log(0) / div-by-tiny if model collapsed).
        var imag_var = sum_sq / Float64(EVAL_BATCH)
        if imag_var < 1e-9:
            imag_var = 1e-9

        var true_mean = mc_mean[t]
        var true_var = mc_var[t]
        if true_var < 1e-9:
            true_var = 1e-9

        # KL(N(true) || N(model))
        var mu_diff_m = true_mean - imag_mean
        var kl_model = (
            0.5 * log(imag_var / true_var)
            + (true_var + mu_diff_m * mu_diff_m) / (2.0 * imag_var)
            - 0.5
        )

        # Persistence: marginal predictive distribution at step t (random ±1 actions).
        var persist_mean = 0.1 * Float64(t - 1)  # ACTION_STEP * (t-1) but accounting for fixed +1 in eval
        # Note: eval used fixed +1 so true_mean = 0.1·t. Persistence (predicting prev step)
        # would mean predicting 0.1·(t-1). Persistence variance combines action + accumulated noise.
        var persist_var = var_action + Float64(t - 1) * var_env_step
        if persist_var < 1e-9:
            persist_var = 1e-9
        var mu_diff_p = true_mean - persist_mean
        var kl_persist = (
            0.5 * log(persist_var / true_var)
            + (true_var + mu_diff_p * mu_diff_p) / (2.0 * persist_var)
            - 0.5
        )

        kl_model_total += kl_model
        kl_persist_total += kl_persist

        print(
            "    ", t,
            "    ", fit(String(kl_model), 9),
            "       ", fit(String(kl_persist), 9),
            "       ", fit(String(kl_persist - kl_model), 9),
        )

    var avg_kl_model = kl_model_total / Float64(SEQ_LEN)
    var avg_kl_persist = kl_persist_total / Float64(SEQ_LEN)

    print("\n  avg KL(env || model)       :", avg_kl_model, "nats")
    print("  avg KL(env || persistence) :", avg_kl_persist, "nats")
    print("  improvement vs persistence :", avg_kl_persist - avg_kl_model, "nats (lower KL = better)")
    var ratio_kl = avg_kl_persist / avg_kl_model if avg_kl_model > 1e-12 else 999.0
    print("  ratio (persist / model)    :", ratio_kl, "× (≥1 means model beats persistence)")

    var pass_mean = avg_mean_rel < 0.10
    var pass_var = avg_var_rel < 0.30
    var pass_kl = avg_kl_model < avg_kl_persist  # must beat persistence

    if pass_mean and pass_var and pass_kl:
        print("\n  [PASS] imagined rollouts match MC truth (mean ≤10%, var ≤30%, KL beats persistence)")
    else:
        if not pass_mean:
            print("\n  [FAIL] avg mean rel err", avg_mean_rel, "≥ 0.10")
        if not pass_var:
            print("\n  [FAIL] avg var  rel err", avg_var_rel, "≥ 0.30")
        if not pass_kl:
            print("\n  [FAIL] avg KL(env||model)", avg_kl_model, "≥ persistence", avg_kl_persist)
        raise Error("world-model dynamics test failed")

    # cleanup
    params_buf.free()
    grads_buf.free()
    opt_state_buf.free()
    opt_global_buf.free()
    eval_lat_buf.free()
    eval_mu_eps_raw.free()
    eval_a_below_raw.free()
    eval_z_below_raw.free()
    eval_dx_raw.free()
    eval_noise_raw.free()
    eval_x_in_buf.free()
    eval_y_dummy_buf.free()
    eval_sample_buf.free()
    imagined.free()
    mc_mean.free()
    mc_var.free()
    print("=== Done ===")

from noeira.core.fmt import fit