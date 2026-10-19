"""Pendulum CEM planning — MLP baseline (single-step recon, apples-to-apples
vs PCN). Mirrors `test_mountain_car_mlp_baseline_cem.mojo` with Pendulum
dynamics and the tuned CEM config (H=40, σ₀=1.0, no velocity penalty) used
in the failing PCN+CEM Pendulum run.

Run:
    pixi run mojo run -I . tests/pcn/test_pendulum_mlp_baseline_cem.mojo
"""

from std.math import sqrt, log, cos, sin, tanh, pi
from std.memory import alloc, memset
from std.random.philox import Random as PhiloxRandom
from std.time import perf_counter_ns
from noeira.core.tiles import tile

from noeira.nn.constants import DT as dtype
from noeira.experimental.pcn.pc_optimizer import PCAdam
from noeira.experimental.pcn.pc_scheduler import CosineWarmupSchedule
from noeira.experimental.pcn import PCEncoder, clip_grad_norm


# Pendulum physics (Gymnasium defaults).
comptime PEND_G: Float64 = 10.0
comptime PEND_L: Float64 = 1.0
comptime PEND_M: Float64 = 1.0
comptime PEND_DT: Float64 = 0.05
comptime PEND_MAX_SPEED: Float64 = 8.0
comptime PEND_MAX_TORQUE: Float64 = 2.0

# Architecture — identical to PCN+MLP-other Pendulum tests (12,930 params).
comptime BATCH = 32
comptime HIDDEN = 64
comptime ACTION_DIM = 1
comptime OBS_DIM = 3
comptime AUG_DIM = HIDDEN + ACTION_DIM
comptime SEQ_LEN = 20
comptime EPOCHS = 100
comptime N_BATCHES_PER_EPOCH = 100
comptime WARMUP_EPOCHS = 5
comptime LR_MIN_SCALE: Float64 = 0.1
comptime ADAM_LR: Float64 = 0.001
comptime GRAD_CLIP_NORM: Float64 = 1.0

comptime ENC_INPUT_DIM = HIDDEN + ACTION_DIM + OBS_DIM
comptime ENC_HIDDEN_DIM = 64
comptime ENC_OUTPUT_DIM = HIDDEN
comptime ENC = PCEncoder[ENC_INPUT_DIM, ENC_HIDDEN_DIM, ENC_OUTPUT_DIM]
comptime ENC_PARAM_SIZE = ENC.PARAM_SIZE

comptime T_PARAM_SIZE = AUG_DIM * HIDDEN + HIDDEN
comptime D_PARAM_SIZE = HIDDEN * OBS_DIM + OBS_DIM
comptime T_W_OFFSET = 0
comptime T_B_OFFSET = AUG_DIM * HIDDEN
comptime D_W_OFFSET = 0
comptime D_B_OFFSET = HIDDEN * OBS_DIM

# CEM planning — tuned for Pendulum (H=40, σ₀=1.0, no velocity penalty).
comptime PLAN_HORIZON = 40
comptime N_SAMPLES = 128
comptime N_ELITES = 16
comptime N_CEM_ITERS = 2
comptime INITIAL_SIGMA: Float64 = 1.0
comptime MIN_SIGMA: Float64 = 0.05
comptime ACTION_PENALTY: Float64 = 0.001
comptime MAX_EPISODE_STEPS = 200
comptime EVAL_WINDOW = 100
comptime UPRIGHT_THRESHOLD: Float64 = 0.9
comptime N_EVAL_EPISODES = 5

comptime OPT = PCAdam[LR=ADAM_LR]
comptime SCHED = CosineWarmupSchedule[
    WARMUP_EPOCHS=WARMUP_EPOCHS, MIN_SCALE=LR_MIN_SCALE
]


def _angle_normalize(t: Float64) -> Float64:
    var x = (t + pi) - 2.0 * pi * Float64(Int((t + pi) / (2.0 * pi)))
    if x < 0.0:
        x += 2.0 * pi
    return x - pi


def _step_pendulum(
    mut theta: Float64, mut theta_dot: Float64, torque: Float64
) -> Tuple[Float64, Float64]:
    var u = torque
    if u > PEND_MAX_TORQUE:
        u = PEND_MAX_TORQUE
    elif u < -PEND_MAX_TORQUE:
        u = -PEND_MAX_TORQUE
    var theta_acc = (3.0 * PEND_G) / (2.0 * PEND_L) * sin(theta) + (
        3.0 / (PEND_M * PEND_L * PEND_L)
    ) * u
    var new_dot = theta_dot + theta_acc * PEND_DT
    if new_dot > PEND_MAX_SPEED:
        new_dot = PEND_MAX_SPEED
    elif new_dot < -PEND_MAX_SPEED:
        new_dot = -PEND_MAX_SPEED
    var new_theta = _angle_normalize(theta + new_dot * PEND_DT)
    return (new_theta, new_dot)


def _gen_rollout_into[
    SEQ_LEN_T: Int
](
    mut rng: PhiloxRandom,
    actions_buf: Pointer[Scalar[dtype], origin=MutAnyOrigin],
    obs_buf: Pointer[Scalar[dtype], origin=MutAnyOrigin],
    actions_offset: Int,
    obs_offset: Int,
):
    var u0 = Float64(rng.step_uniform()[0])
    var u1 = Float64(rng.step_uniform()[0])
    var theta = (u0 * 2.0 - 1.0) * pi
    var theta_dot = (u1 * 2.0 - 1.0) * 1.0
    obs_buf[obs_offset + 0] = Scalar[dtype](cos(theta))
    obs_buf[obs_offset + 1] = Scalar[dtype](sin(theta))
    obs_buf[obs_offset + 2] = Scalar[dtype](theta_dot / PEND_MAX_SPEED)
    for t in range(SEQ_LEN_T):
        var ua = Float64(rng.step_uniform()[0])
        var torque_norm = ua * 2.0 - 1.0
        var torque = torque_norm * PEND_MAX_TORQUE
        actions_buf[actions_offset + t] = Scalar[dtype](torque_norm)
        var stepped = _step_pendulum(theta, theta_dot, torque)
        theta = stepped[0]
        theta_dot = stepped[1]
        obs_buf[obs_offset + (t + 1) * OBS_DIM + 0] = Scalar[dtype](cos(theta))
        obs_buf[obs_offset + (t + 1) * OBS_DIM + 1] = Scalar[dtype](sin(theta))
        obs_buf[obs_offset + (t + 1) * OBS_DIM + 2] = Scalar[dtype](
            theta_dot / PEND_MAX_SPEED
        )


def _gauss_pair(mut rng: PhiloxRandom) -> Tuple[Float64, Float64]:
    var u1 = Float64(rng.step_uniform()[0])
    var u2 = Float64(rng.step_uniform()[0])
    if u1 < 1e-12:
        u1 = 1e-12
    var r = sqrt(-2.0 * log(u1))
    var theta = 2.0 * pi * u2
    return (r * cos(theta), r * sin(theta))


def _xavier_init_layer(
    params: Pointer[Scalar[dtype], origin=MutAnyOrigin],
    in_dim: Int,
    out_dim: Int,
    seed: UInt64,
):
    var rng = PhiloxRandom(seed=seed, offset=UInt64(0))
    var bound = sqrt(Float64(6.0) / Float64(in_dim + out_dim))
    for i in range(in_dim * out_dim):
        var u = Float64(rng.step_uniform()[0])
        params[i] = Scalar[dtype]((u * 2.0 - 1.0) * bound)
    for j in range(out_dim):
        params[in_dim * out_dim + j] = Scalar[dtype](0.0)


def _lt_forward[
    BATCH_T: Int, IN: Int, OUT: Int
](
    params: Pointer[Scalar[dtype], origin=MutAnyOrigin],
    x: Pointer[Scalar[dtype], origin=MutAnyOrigin],
    a: Pointer[Scalar[dtype], origin=MutAnyOrigin],
    mu: Pointer[Scalar[dtype], origin=MutAnyOrigin],
):
    """Forward: a = tanh(x); mu = a @ W + b. Caches `a` for backward."""
    for s in range(BATCH_T):
        for i in range(IN):
            a[s * IN + i] = Scalar[dtype](tanh(Float64(x[s * IN + i])))
        for j in range(OUT):
            var sum_j = Float64(params[IN * OUT + j])
            for i in range(IN):
                sum_j += Float64(a[s * IN + i]) * Float64(params[i * OUT + j])
            mu[s * OUT + j] = Scalar[dtype](sum_j)


def _lt_backward[
    BATCH_T: Int, IN: Int, OUT: Int
](
    params: Pointer[Scalar[dtype], origin=MutAnyOrigin],
    a: Pointer[Scalar[dtype], origin=MutAnyOrigin],
    d_mu: Pointer[Scalar[dtype], origin=MutAnyOrigin],
    d_W: Pointer[Scalar[dtype], origin=MutAnyOrigin],
    d_b: Pointer[Scalar[dtype], origin=MutAnyOrigin],
    d_x: Pointer[Scalar[dtype], origin=MutAnyOrigin],
):
    """Backward through `μ = tanh(x) @ W + b` (overwrites all outputs)."""
    for k in range(IN * OUT):
        d_W[k] = Scalar[dtype](0.0)
    for j in range(OUT):
        d_b[j] = Scalar[dtype](0.0)
    for s in range(BATCH_T):
        for j in range(OUT):
            d_b[j] = Scalar[dtype](Float64(d_b[j]) + Float64(d_mu[s * OUT + j]))
        for i in range(IN):
            var a_i = Float64(a[s * IN + i])
            for j in range(OUT):
                var idx = i * OUT + j
                d_W[idx] = Scalar[dtype](
                    Float64(d_W[idx]) + a_i * Float64(d_mu[s * OUT + j])
                )
        for i in range(IN):
            var d_a_i: Float64 = 0
            for j in range(OUT):
                d_a_i += Float64(params[i * OUT + j]) * Float64(
                    d_mu[s * OUT + j]
                )
            var a_i = Float64(a[s * IN + i])
            d_x[s * IN + i] = Scalar[dtype](d_a_i * (1.0 - a_i * a_i))


def main() raises:
    print("=" * 60)
    print("Pendulum — MLP baseline (single-step recon) + CEM planner")
    print("=" * 60)
    print("  Arch       : Encoder MLP (", ENC.PARAM_SIZE, " params)")
    print(
        "              + Linear[",
        AUG_DIM,
        "→",
        HIDDEN,
        "]+tanh transition (",
        T_PARAM_SIZE,
        ")",
    )
    print(
        "              + Linear[",
        HIDDEN,
        "→",
        OBS_DIM,
        "]+tanh decoder (",
        D_PARAM_SIZE,
        ")",
    )
    print(
        "  Total      :",
        ENC.PARAM_SIZE + T_PARAM_SIZE + D_PARAM_SIZE,
        " params",
    )
    print("  Training   : single-step reconstruction, end-to-end backprop")
    print(
        "  CEM        : H=",
        PLAN_HORIZON,
        " N=",
        N_SAMPLES,
        " K=",
        N_ELITES,
        " iters=",
        N_CEM_ITERS,
    )
    print("  Score      : Σ cos θ - 0.001·a²  (no velocity penalty)")
    print(
        "  Pass       : avg cos θ over last",
        EVAL_WINDOW,
        " > ",
        UPRIGHT_THRESHOLD,
    )

    # ── Model params + Adam state ─────────────────────────────────────────────
    var T_params_buf = alloc[Scalar[dtype]](T_PARAM_SIZE).as_unsafe_any_origin()
    var T_grads_buf = alloc[Scalar[dtype]](T_PARAM_SIZE).as_unsafe_any_origin()
    var T_opt_state_buf = alloc[Scalar[dtype]](
        T_PARAM_SIZE * OPT.STATE_PER_PARAM
    ).as_unsafe_any_origin()
    var T_opt_global_buf = alloc[Scalar[dtype]](OPT.GLOBAL_STATE_SIZE).as_unsafe_any_origin()
    memset(T_params_buf, 0, T_PARAM_SIZE)
    memset(T_grads_buf, 0, T_PARAM_SIZE)
    memset(T_opt_state_buf, 0, T_PARAM_SIZE * OPT.STATE_PER_PARAM)
    memset(T_opt_global_buf, 0, OPT.GLOBAL_STATE_SIZE)
    _xavier_init_layer(T_params_buf, AUG_DIM, HIDDEN, UInt64(7))
    var T_params = tile[T_PARAM_SIZE](T_params_buf)
    var T_grads = tile[T_PARAM_SIZE](T_grads_buf)
    var T_opt_state = tile[T_PARAM_SIZE, OPT.STATE_PER_PARAM](T_opt_state_buf)
    var T_opt_global = tile[OPT.GLOBAL_STATE_SIZE](T_opt_global_buf)

    var D_params_buf = alloc[Scalar[dtype]](D_PARAM_SIZE).as_unsafe_any_origin()
    var D_grads_buf = alloc[Scalar[dtype]](D_PARAM_SIZE).as_unsafe_any_origin()
    var D_opt_state_buf = alloc[Scalar[dtype]](
        D_PARAM_SIZE * OPT.STATE_PER_PARAM
    ).as_unsafe_any_origin()
    var D_opt_global_buf = alloc[Scalar[dtype]](OPT.GLOBAL_STATE_SIZE).as_unsafe_any_origin()
    memset(D_params_buf, 0, D_PARAM_SIZE)
    memset(D_grads_buf, 0, D_PARAM_SIZE)
    memset(D_opt_state_buf, 0, D_PARAM_SIZE * OPT.STATE_PER_PARAM)
    memset(D_opt_global_buf, 0, OPT.GLOBAL_STATE_SIZE)
    _xavier_init_layer(D_params_buf, HIDDEN, OBS_DIM, UInt64(8))
    var D_params = tile[D_PARAM_SIZE](D_params_buf)
    var D_grads = tile[D_PARAM_SIZE](D_grads_buf)
    var D_opt_state = tile[D_PARAM_SIZE, OPT.STATE_PER_PARAM](D_opt_state_buf)
    var D_opt_global = tile[OPT.GLOBAL_STATE_SIZE](D_opt_global_buf)

    var enc_params_buf = alloc[Scalar[dtype]](ENC_PARAM_SIZE).as_unsafe_any_origin()
    var enc_grads_buf = alloc[Scalar[dtype]](ENC_PARAM_SIZE).as_unsafe_any_origin()
    var enc_opt_state_buf = alloc[Scalar[dtype]](
        ENC_PARAM_SIZE * OPT.STATE_PER_PARAM
    ).as_unsafe_any_origin()
    var enc_opt_global_buf = alloc[Scalar[dtype]](OPT.GLOBAL_STATE_SIZE).as_unsafe_any_origin()
    memset(enc_params_buf, 0, ENC_PARAM_SIZE)
    memset(enc_grads_buf, 0, ENC_PARAM_SIZE)
    memset(enc_opt_state_buf, 0, ENC_PARAM_SIZE * OPT.STATE_PER_PARAM)
    memset(enc_opt_global_buf, 0, OPT.GLOBAL_STATE_SIZE)
    var enc_params = tile[ENC_PARAM_SIZE](enc_params_buf)
    var enc_grads = tile[ENC_PARAM_SIZE](enc_grads_buf)
    var enc_opt_state = tile[ENC_PARAM_SIZE, OPT.STATE_PER_PARAM](enc_opt_state_buf)
    var enc_opt_global = tile[OPT.GLOBAL_STATE_SIZE](enc_opt_global_buf)
    ENC.xavier_init[dtype](enc_params, UInt64(123))

    # ── Training scratch (BATCH=32) ───────────────────────────────────────────
    var enc_input_buf = alloc[Scalar[dtype]](BATCH * ENC_INPUT_DIM).as_unsafe_any_origin()
    var enc_hpre_buf = alloc[Scalar[dtype]](BATCH * ENC_HIDDEN_DIM).as_unsafe_any_origin()
    var enc_hact_buf = alloc[Scalar[dtype]](BATCH * ENC_HIDDEN_DIM).as_unsafe_any_origin()
    var enc_output_buf = alloc[Scalar[dtype]](BATCH * ENC_OUTPUT_DIM).as_unsafe_any_origin()
    var enc_dz_buf = alloc[Scalar[dtype]](BATCH * ENC_OUTPUT_DIM).as_unsafe_any_origin()
    var enc_input = tile[BATCH, ENC_INPUT_DIM](enc_input_buf)
    var enc_hpre = tile[BATCH, ENC_HIDDEN_DIM](enc_hpre_buf)
    var enc_hact = tile[BATCH, ENC_HIDDEN_DIM](enc_hact_buf)
    var enc_output = tile[BATCH, ENC_OUTPUT_DIM](enc_output_buf)
    var enc_dz = tile[BATCH, ENC_OUTPUT_DIM](enc_dz_buf)

    var x_aug_buf = alloc[Scalar[dtype]](BATCH * AUG_DIM).as_unsafe_any_origin()
    var a_x_aug_buf = alloc[Scalar[dtype]](BATCH * AUG_DIM).as_unsafe_any_origin()
    var mu_z_next_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var a_z_next_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var mu_obs_buf = alloc[Scalar[dtype]](BATCH * OBS_DIM).as_unsafe_any_origin()
    var d_mu_obs_buf = alloc[Scalar[dtype]](BATCH * OBS_DIM).as_unsafe_any_origin()
    var d_mu_z_next_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var d_x_aug_buf = alloc[Scalar[dtype]](BATCH * AUG_DIM).as_unsafe_any_origin()

    var actions_buf = alloc[Scalar[dtype]](BATCH * SEQ_LEN).as_unsafe_any_origin()
    var obs_buf = alloc[Scalar[dtype]](BATCH * (SEQ_LEN + 1) * OBS_DIM).as_unsafe_any_origin()

    # ── Train ────────────────────────────────────────────────────────────────
    print("\n  epoch | last_step_loss | wall_t (s)")
    print("  ------+----------------+------------")
    var rng = PhiloxRandom(seed=UInt64(7), offset=UInt64(0))
    var step_num: Int = 0
    var prev_z_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var t0 = perf_counter_ns()

    for epoch in range(EPOCHS):
        var lr_scale = SCHED.lr_scale_at(epoch, EPOCHS)
        var last_loss: Float64 = 0.0
        for batch_idx in range(N_BATCHES_PER_EPOCH):
            for b in range(BATCH):
                _gen_rollout_into[SEQ_LEN](
                    rng,
                    actions_buf,
                    obs_buf,
                    b * SEQ_LEN,
                    b * (SEQ_LEN + 1) * OBS_DIM,
                )
            memset(prev_z_buf, 0, BATCH * HIDDEN)
            for t in range(0, SEQ_LEN):
                for b in range(BATCH):
                    for j in range(HIDDEN):
                        enc_input_buf[b * ENC_INPUT_DIM + j] = prev_z_buf[
                            b * HIDDEN + j
                        ]
                    var prev_action = Scalar[dtype](0.0) if t == 0 else (
                        actions_buf[b * SEQ_LEN + (t - 1)]
                    )
                    enc_input_buf[b * ENC_INPUT_DIM + HIDDEN] = prev_action
                    for d in range(OBS_DIM):
                        enc_input_buf[
                            b * ENC_INPUT_DIM + HIDDEN + ACTION_DIM + d
                        ] = obs_buf[
                            b * (SEQ_LEN + 1) * OBS_DIM + t * OBS_DIM + d
                        ]

                ENC.forward[BATCH, dtype](
                    enc_params, enc_input, enc_hpre, enc_hact, enc_output
                )

                for b in range(BATCH):
                    for j in range(HIDDEN):
                        x_aug_buf[b * AUG_DIM + j] = enc_output_buf[
                            b * ENC_OUTPUT_DIM + j
                        ]
                    x_aug_buf[b * AUG_DIM + HIDDEN] = actions_buf[
                        b * SEQ_LEN + t
                    ]

                _lt_forward[BATCH, AUG_DIM, HIDDEN](
                    T_params_buf, x_aug_buf, a_x_aug_buf, mu_z_next_buf
                )
                _lt_forward[BATCH, HIDDEN, OBS_DIM](
                    D_params_buf, mu_z_next_buf, a_z_next_buf, mu_obs_buf
                )

                var batch_loss: Float64 = 0.0
                for b in range(BATCH):
                    for d in range(OBS_DIM):
                        var diff = Float64(
                            mu_obs_buf[b * OBS_DIM + d]
                        ) - Float64(
                            obs_buf[
                                b * (SEQ_LEN + 1) * OBS_DIM
                                + (t + 1) * OBS_DIM
                                + d
                            ]
                        )
                        d_mu_obs_buf[b * OBS_DIM + d] = Scalar[dtype](diff)
                        batch_loss += 0.5 * diff * diff
                last_loss = batch_loss / Float64(BATCH)

                _lt_backward[BATCH, HIDDEN, OBS_DIM](
                    D_params_buf,
                    a_z_next_buf,
                    d_mu_obs_buf,
                    D_grads_buf + D_W_OFFSET,
                    D_grads_buf + D_B_OFFSET,
                    d_mu_z_next_buf,
                )
                _lt_backward[BATCH, AUG_DIM, HIDDEN](
                    T_params_buf,
                    a_x_aug_buf,
                    d_mu_z_next_buf,
                    T_grads_buf + T_W_OFFSET,
                    T_grads_buf + T_B_OFFSET,
                    d_x_aug_buf,
                )

                for b in range(BATCH):
                    for j in range(HIDDEN):
                        enc_dz_buf[b * ENC_OUTPUT_DIM + j] = d_x_aug_buf[
                            b * AUG_DIM + j
                        ]
                ENC.backward[BATCH, dtype](
                    enc_params, enc_input, enc_hact, enc_dz, enc_grads
                )

                clip_grad_norm[T_PARAM_SIZE, dtype](T_grads, GRAD_CLIP_NORM)
                clip_grad_norm[D_PARAM_SIZE, dtype](D_grads, GRAD_CLIP_NORM)
                clip_grad_norm[ENC_PARAM_SIZE, dtype](enc_grads, GRAD_CLIP_NORM)
                step_num += 1
                OPT.step[T_PARAM_SIZE, dtype](
                    T_params,
                    T_grads,
                    T_opt_state,
                    T_opt_global,
                    step_num,
                    lr_scale=lr_scale,
                )
                OPT.step[D_PARAM_SIZE, dtype](
                    D_params,
                    D_grads,
                    D_opt_state,
                    D_opt_global,
                    step_num,
                    lr_scale=lr_scale,
                )
                OPT.step[ENC_PARAM_SIZE, dtype](
                    enc_params,
                    enc_grads,
                    enc_opt_state,
                    enc_opt_global,
                    step_num,
                    lr_scale=lr_scale,
                )

                for b in range(BATCH):
                    for j in range(HIDDEN):
                        prev_z_buf[b * HIDDEN + j] = enc_output_buf[
                            b * ENC_OUTPUT_DIM + j
                        ]

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            var elapsed = Float64(perf_counter_ns() - t0) / 1e9
            print(
                "    ep=",
                epoch,
                "  loss=",
                last_loss,
                "  lr_scale=",
                lr_scale,
                "  wall=",
                elapsed,
                "s",
            )

    var total_t = Float64(perf_counter_ns() - t0) / 1e9
    print("\n  total train time:", total_t, "s")

    # ── CEM imagination scratch (BATCH=N_SAMPLES) ─────────────────────────────
    var cem_z_buf = alloc[Scalar[dtype]](N_SAMPLES * HIDDEN).as_unsafe_any_origin()
    var cem_x_aug_buf = alloc[Scalar[dtype]](N_SAMPLES * AUG_DIM).as_unsafe_any_origin()
    var cem_a_x_aug_buf = alloc[Scalar[dtype]](N_SAMPLES * AUG_DIM).as_unsafe_any_origin()
    var cem_mu_z_next_buf = alloc[Scalar[dtype]](N_SAMPLES * HIDDEN).as_unsafe_any_origin()
    var cem_a_z_next_buf = alloc[Scalar[dtype]](N_SAMPLES * HIDDEN).as_unsafe_any_origin()
    var cem_mu_obs_buf = alloc[Scalar[dtype]](N_SAMPLES * OBS_DIM).as_unsafe_any_origin()
    var cem_actions_buf = alloc[Scalar[dtype]](N_SAMPLES * PLAN_HORIZON).as_unsafe_any_origin()

    var cem_mu = List[Float64](capacity=PLAN_HORIZON)
    var cem_sigma = List[Float64](capacity=PLAN_HORIZON)
    var cem_scores = List[Float64](capacity=N_SAMPLES)
    var cem_indices = List[Int](capacity=N_SAMPLES)
    for _ in range(PLAN_HORIZON):
        cem_mu.append(0.0)
        cem_sigma.append(INITIAL_SIGMA)
    for _ in range(N_SAMPLES):
        cem_scores.append(0.0)
        cem_indices.append(0)

    var agent_z_buf = alloc[Scalar[dtype]](HIDDEN).as_unsafe_any_origin()
    var ep_cos_history = List[Float64](capacity=MAX_EPISODE_STEPS)
    for _ in range(MAX_EPISODE_STEPS):
        ep_cos_history.append(0.0)

    print("\n  === CEM planning evaluation (MLP single-step baseline) ===")
    var eval_rng = PhiloxRandom(seed=UInt64(2027), offset=UInt64(0))
    var n_success: Int = 0
    var t_eval_start = perf_counter_ns()

    var enc_input_1 = tile[1, ENC_INPUT_DIM](enc_input_buf)
    var enc_hpre_1 = tile[1, ENC_HIDDEN_DIM](enc_hpre_buf)
    var enc_hact_1 = tile[1, ENC_HIDDEN_DIM](enc_hact_buf)
    var enc_output_1 = tile[1, ENC_OUTPUT_DIM](enc_output_buf)

    for ep in range(N_EVAL_EPISODES):
        var u0 = Float64(eval_rng.step_uniform()[0])
        var u1 = Float64(eval_rng.step_uniform()[0])
        var theta = (u0 * 2.0 - 1.0) * pi
        var theta_dot = (u1 * 2.0 - 1.0) * 1.0

        memset(agent_z_buf, 0, HIDDEN)
        for h in range(PLAN_HORIZON):
            cem_mu[h] = 0.0
            cem_sigma[h] = INITIAL_SIGMA

        for j in range(HIDDEN):
            enc_input_buf[j] = Scalar[dtype](0.0)
        enc_input_buf[HIDDEN] = Scalar[dtype](0.0)
        enc_input_buf[HIDDEN + ACTION_DIM + 0] = Scalar[dtype](cos(theta))
        enc_input_buf[HIDDEN + ACTION_DIM + 1] = Scalar[dtype](sin(theta))
        enc_input_buf[HIDDEN + ACTION_DIM + 2] = Scalar[dtype](
            theta_dot / PEND_MAX_SPEED
        )
        ENC.forward[1, dtype](
            enc_params, enc_input_1, enc_hpre_1, enc_hact_1, enc_output_1
        )
        for j in range(HIDDEN):
            agent_z_buf[j] = enc_output_buf[j]

        for step in range(MAX_EPISODE_STEPS):
            ep_cos_history[step] = cos(theta)

            for cem_iter in range(N_CEM_ITERS):
                var total = N_SAMPLES * PLAN_HORIZON
                var i = 0
                while i < total:
                    var pair = _gauss_pair(eval_rng)
                    var s0 = i // PLAN_HORIZON
                    var h0 = i % PLAN_HORIZON
                    var a0 = cem_mu[h0] + cem_sigma[h0] * pair[0]
                    if a0 > 1.0:
                        a0 = 1.0
                    elif a0 < -1.0:
                        a0 = -1.0
                    cem_actions_buf[s0 * PLAN_HORIZON + h0] = Scalar[dtype](a0)
                    i += 1
                    if i < total:
                        var s1 = i // PLAN_HORIZON
                        var h1 = i % PLAN_HORIZON
                        var a1 = cem_mu[h1] + cem_sigma[h1] * pair[1]
                        if a1 > 1.0:
                            a1 = 1.0
                        elif a1 < -1.0:
                            a1 = -1.0
                        cem_actions_buf[s1 * PLAN_HORIZON + h1] = Scalar[dtype](
                            a1
                        )
                        i += 1

                for s in range(N_SAMPLES):
                    for j in range(HIDDEN):
                        cem_z_buf[s * HIDDEN + j] = agent_z_buf[j]
                    cem_scores[s] = 0.0

                for h in range(PLAN_HORIZON):
                    for s in range(N_SAMPLES):
                        for j in range(HIDDEN):
                            cem_x_aug_buf[s * AUG_DIM + j] = cem_z_buf[
                                s * HIDDEN + j
                            ]
                        cem_x_aug_buf[s * AUG_DIM + HIDDEN] = cem_actions_buf[
                            s * PLAN_HORIZON + h
                        ]
                    _lt_forward[N_SAMPLES, AUG_DIM, HIDDEN](
                        T_params_buf,
                        cem_x_aug_buf,
                        cem_a_x_aug_buf,
                        cem_mu_z_next_buf,
                    )
                    _lt_forward[N_SAMPLES, HIDDEN, OBS_DIM](
                        D_params_buf,
                        cem_mu_z_next_buf,
                        cem_a_z_next_buf,
                        cem_mu_obs_buf,
                    )
                    for s in range(N_SAMPLES):
                        var cos_pred = Float64(cem_mu_obs_buf[s * OBS_DIM + 0])
                        var a = Float64(cem_actions_buf[s * PLAN_HORIZON + h])
                        cem_scores[s] += cos_pred - ACTION_PENALTY * a * a
                    for s in range(N_SAMPLES):
                        for j in range(HIDDEN):
                            cem_z_buf[s * HIDDEN + j] = cem_mu_z_next_buf[
                                s * HIDDEN + j
                            ]

                for s in range(N_SAMPLES):
                    cem_indices[s] = s
                for k in range(N_ELITES):
                    var best_s = k
                    var best_score = cem_scores[cem_indices[k]]
                    for s in range(k + 1, N_SAMPLES):
                        if cem_scores[cem_indices[s]] > best_score:
                            best_score = cem_scores[cem_indices[s]]
                            best_s = s
                    if best_s != k:
                        var tmp = cem_indices[k]
                        cem_indices[k] = cem_indices[best_s]
                        cem_indices[best_s] = tmp

                for h in range(PLAN_HORIZON):
                    var s_mu: Float64 = 0
                    for k in range(N_ELITES):
                        s_mu += Float64(
                            cem_actions_buf[cem_indices[k] * PLAN_HORIZON + h]
                        )
                    var new_mu = s_mu / Float64(N_ELITES)
                    var s_var: Float64 = 0
                    for k in range(N_ELITES):
                        var d = (
                            Float64(
                                cem_actions_buf[
                                    cem_indices[k] * PLAN_HORIZON + h
                                ]
                            )
                            - new_mu
                        )
                        s_var += d * d
                    var new_sigma = sqrt(s_var / Float64(N_ELITES))
                    if new_sigma < MIN_SIGMA:
                        new_sigma = MIN_SIGMA
                    cem_mu[h] = new_mu
                    cem_sigma[h] = new_sigma

            var action_norm = cem_mu[0]
            if action_norm > 1.0:
                action_norm = 1.0
            elif action_norm < -1.0:
                action_norm = -1.0
            var torque = action_norm * PEND_MAX_TORQUE
            var stepped = _step_pendulum(theta, theta_dot, torque)
            theta = stepped[0]
            theta_dot = stepped[1]

            for j in range(HIDDEN):
                enc_input_buf[j] = agent_z_buf[j]
            enc_input_buf[HIDDEN] = Scalar[dtype](action_norm)
            enc_input_buf[HIDDEN + ACTION_DIM + 0] = Scalar[dtype](cos(theta))
            enc_input_buf[HIDDEN + ACTION_DIM + 1] = Scalar[dtype](sin(theta))
            enc_input_buf[HIDDEN + ACTION_DIM + 2] = Scalar[dtype](
                theta_dot / PEND_MAX_SPEED
            )
            ENC.forward[1, dtype](
                enc_params, enc_input_1, enc_hpre_1, enc_hact_1, enc_output_1
            )
            for j in range(HIDDEN):
                agent_z_buf[j] = enc_output_buf[j]

            for h in range(PLAN_HORIZON - 1):
                cem_mu[h] = cem_mu[h + 1]
                cem_sigma[h] = cem_sigma[h + 1]
            cem_mu[PLAN_HORIZON - 1] = 0.0
            cem_sigma[PLAN_HORIZON - 1] = INITIAL_SIGMA

        var avg_cos: Float64 = 0
        var win_start = MAX_EPISODE_STEPS - EVAL_WINDOW
        for s in range(win_start, MAX_EPISODE_STEPS):
            avg_cos += ep_cos_history[s]
        avg_cos /= Float64(EVAL_WINDOW)

        var passed = avg_cos > UPRIGHT_THRESHOLD
        if passed:
            n_success += 1
        print(
            "    ep=",
            ep,
            " : avg cos θ (last ",
            EVAL_WINDOW,
            ") =",
            avg_cos,
            "  →",
            "PASS" if passed else "MISS",
        )

    var t_eval = Float64(perf_counter_ns() - t_eval_start) / 1e9
    print("\n  eval wall time:", t_eval, "s")
    print("  success rate :", n_success, "/", N_EVAL_EPISODES)

    var pass_threshold = (N_EVAL_EPISODES + 1) // 2
    if n_success >= pass_threshold:
        print(
            "\n  [PASS] CEM planner solved",
            n_success,
            "/",
            N_EVAL_EPISODES,
            " (threshold:",
            pass_threshold,
            ")",
        )
    else:
        print(
            "\n  [FAIL] CEM planner solved only",
            n_success,
            "/",
            N_EVAL_EPISODES,
            " (need ≥",
            pass_threshold,
            ")",
        )

    # cleanup
    T_params_buf.free()
    T_grads_buf.free()
    T_opt_state_buf.free()
    T_opt_global_buf.free()
    D_params_buf.free()
    D_grads_buf.free()
    D_opt_state_buf.free()
    D_opt_global_buf.free()
    enc_params_buf.free()
    enc_grads_buf.free()
    enc_opt_state_buf.free()
    enc_opt_global_buf.free()
    enc_input_buf.free()
    enc_hpre_buf.free()
    enc_hact_buf.free()
    enc_output_buf.free()
    enc_dz_buf.free()
    x_aug_buf.free()
    a_x_aug_buf.free()
    mu_z_next_buf.free()
    a_z_next_buf.free()
    mu_obs_buf.free()
    d_mu_obs_buf.free()
    d_mu_z_next_buf.free()
    d_x_aug_buf.free()
    actions_buf.free()
    obs_buf.free()
    prev_z_buf.free()
    cem_z_buf.free()
    cem_x_aug_buf.free()
    cem_a_x_aug_buf.free()
    cem_mu_z_next_buf.free()
    cem_a_z_next_buf.free()
    cem_mu_obs_buf.free()
    cem_actions_buf.free()
    agent_z_buf.free()
    print("=== Done ===")
