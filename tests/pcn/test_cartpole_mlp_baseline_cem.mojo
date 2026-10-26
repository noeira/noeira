"""CartPole-Continuous CEM planning — MLP baseline (single-step recon).

Mirror of `test_cartpole_cem_planning.mojo` with PCN training swapped for
end-to-end reconstruction-loss backprop. Same architecture, same CEM, same
eval seed.

Run:
    pixi run mojo run -I . tests/pcn/test_cartpole_mlp_baseline_cem.mojo
"""

from std.math import sqrt, log, cos, sin, tanh, pi
from std.memory import alloc, unsafe_memset
from std.random.philox import Random as PhiloxRandom
from std.time import perf_counter_ns
from noeira.core.tiles import tile

from noeira.nn.constants import DT as dtype
from noeira.experimental.pcn.pc_optimizer import PCAdam
from noeira.experimental.pcn.pc_scheduler import CosineWarmupSchedule
from noeira.experimental.pcn import PCEncoder, clip_grad_norm


# CartPole-Continuous physics (Gymnasium defaults).
comptime CP_GRAVITY: Float64 = 9.8
comptime CP_MASSCART: Float64 = 1.0
comptime CP_MASSPOLE: Float64 = 0.1
comptime CP_TOTAL_MASS: Float64 = CP_MASSCART + CP_MASSPOLE
comptime CP_LENGTH: Float64 = 0.5
comptime CP_POLEMASS_LENGTH: Float64 = CP_MASSPOLE * CP_LENGTH
comptime CP_FORCE_MAG: Float64 = 10.0
comptime CP_TAU: Float64 = 0.02
comptime CP_X_THRESHOLD: Float64 = 2.4
comptime CP_THETA_THRESHOLD: Float64 = 0.2
comptime CP_RESET_RANGE: Float64 = 0.05
comptime CP_X_SCALE: Float64 = 2.4
comptime CP_XDOT_SCALE: Float64 = 3.0
comptime CP_THETA_SCALE: Float64 = 0.2
comptime CP_THETADOT_SCALE: Float64 = 2.0

# Architecture.
comptime BATCH = 32
comptime HIDDEN = 64
comptime ACTION_DIM = 1
comptime OBS_DIM = 4
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

# CEM planning.
comptime PLAN_HORIZON = 20
comptime N_SAMPLES = 128
comptime N_ELITES = 16
comptime N_CEM_ITERS = 2
comptime INITIAL_SIGMA: Float64 = 0.5
comptime MIN_SIGMA: Float64 = 0.05
comptime ACTION_PENALTY: Float64 = 0.001
comptime POS_PENALTY: Float64 = 0.05
comptime MAX_EPISODE_STEPS = 200
comptime PASS_STEPS = 195
comptime N_EVAL_EPISODES = 5

comptime OPT = PCAdam[LR=ADAM_LR]
comptime SCHED = CosineWarmupSchedule[
    WARMUP_EPOCHS=WARMUP_EPOCHS, MIN_SCALE=LR_MIN_SCALE
]


def _step_cartpole(
    x: Float64,
    x_dot: Float64,
    theta: Float64,
    theta_dot: Float64,
    action_norm: Float64,
) -> Tuple[Float64, Float64, Float64, Float64]:
    var u = action_norm
    if u > 1.0:
        u = 1.0
    elif u < -1.0:
        u = -1.0
    var force = u * CP_FORCE_MAG
    var ct = cos(theta)
    var st = sin(theta)
    var temp = (
        force + CP_POLEMASS_LENGTH * theta_dot * theta_dot * st
    ) / CP_TOTAL_MASS
    var theta_acc = (CP_GRAVITY * st - ct * temp) / (
        CP_LENGTH * (4.0 / 3.0 - CP_MASSPOLE * ct * ct / CP_TOTAL_MASS)
    )
    var x_acc = temp - CP_POLEMASS_LENGTH * theta_acc * ct / CP_TOTAL_MASS
    return (
        x + CP_TAU * x_dot,
        x_dot + CP_TAU * x_acc,
        theta + CP_TAU * theta_dot,
        theta_dot + CP_TAU * theta_acc,
    )


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
    var u2 = Float64(rng.step_uniform()[0])
    var u3 = Float64(rng.step_uniform()[0])
    var x = (u0 * 2.0 - 1.0) * CP_RESET_RANGE
    var x_dot = (u1 * 2.0 - 1.0) * CP_RESET_RANGE
    var theta = (u2 * 2.0 - 1.0) * CP_RESET_RANGE
    var theta_dot = (u3 * 2.0 - 1.0) * CP_RESET_RANGE
    obs_buf[obs_offset + 0] = Scalar[dtype](x / CP_X_SCALE)
    obs_buf[obs_offset + 1] = Scalar[dtype](x_dot / CP_XDOT_SCALE)
    obs_buf[obs_offset + 2] = Scalar[dtype](theta / CP_THETA_SCALE)
    obs_buf[obs_offset + 3] = Scalar[dtype](theta_dot / CP_THETADOT_SCALE)
    for t in range(SEQ_LEN_T):
        var ua = Float64(rng.step_uniform()[0])
        var action_norm = ua * 2.0 - 1.0
        actions_buf[actions_offset + t] = Scalar[dtype](action_norm)
        var stepped = _step_cartpole(x, x_dot, theta, theta_dot, action_norm)
        x = stepped[0]
        x_dot = stepped[1]
        theta = stepped[2]
        theta_dot = stepped[3]
        obs_buf[obs_offset + (t + 1) * OBS_DIM + 0] = Scalar[dtype](
            x / CP_X_SCALE
        )
        obs_buf[obs_offset + (t + 1) * OBS_DIM + 1] = Scalar[dtype](
            x_dot / CP_XDOT_SCALE
        )
        obs_buf[obs_offset + (t + 1) * OBS_DIM + 2] = Scalar[dtype](
            theta / CP_THETA_SCALE
        )
        obs_buf[obs_offset + (t + 1) * OBS_DIM + 3] = Scalar[dtype](
            theta_dot / CP_THETADOT_SCALE
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
    print("CartPole-Continuous — MLP baseline (single-step) + CEM")
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
    print("  Score      : -Σ (θ² + 0.05·x² + 0.001·a²)")
    print(
        "  Pass       : survive ≥",
        PASS_STEPS,
        " of",
        MAX_EPISODE_STEPS,
        " steps",
    )

    var T_params_buf = alloc[Scalar[dtype]](T_PARAM_SIZE).as_unsafe_any_origin()
    var T_grads_buf = alloc[Scalar[dtype]](T_PARAM_SIZE).as_unsafe_any_origin()
    var T_opt_state_buf = alloc[Scalar[dtype]](
        T_PARAM_SIZE * OPT.STATE_PER_PARAM
    ).as_unsafe_any_origin()
    var T_opt_global_buf = alloc[Scalar[dtype]](OPT.GLOBAL_STATE_SIZE).as_unsafe_any_origin()
    unsafe_memset(T_params_buf, 0, T_PARAM_SIZE)
    unsafe_memset(T_grads_buf, 0, T_PARAM_SIZE)
    unsafe_memset(T_opt_state_buf, 0, T_PARAM_SIZE * OPT.STATE_PER_PARAM)
    unsafe_memset(T_opt_global_buf, 0, OPT.GLOBAL_STATE_SIZE)
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
    unsafe_memset(D_params_buf, 0, D_PARAM_SIZE)
    unsafe_memset(D_grads_buf, 0, D_PARAM_SIZE)
    unsafe_memset(D_opt_state_buf, 0, D_PARAM_SIZE * OPT.STATE_PER_PARAM)
    unsafe_memset(D_opt_global_buf, 0, OPT.GLOBAL_STATE_SIZE)
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
    unsafe_memset(enc_params_buf, 0, ENC_PARAM_SIZE)
    unsafe_memset(enc_grads_buf, 0, ENC_PARAM_SIZE)
    unsafe_memset(enc_opt_state_buf, 0, ENC_PARAM_SIZE * OPT.STATE_PER_PARAM)
    unsafe_memset(enc_opt_global_buf, 0, OPT.GLOBAL_STATE_SIZE)
    var enc_params = tile[ENC_PARAM_SIZE](enc_params_buf)
    var enc_grads = tile[ENC_PARAM_SIZE](enc_grads_buf)
    var enc_opt_state = tile[ENC_PARAM_SIZE, OPT.STATE_PER_PARAM](enc_opt_state_buf)
    var enc_opt_global = tile[OPT.GLOBAL_STATE_SIZE](enc_opt_global_buf)
    ENC.xavier_init[dtype](enc_params, UInt64(123))

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
            unsafe_memset(prev_z_buf, 0, BATCH * HIDDEN)
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

    # CEM imagination scratch.
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

    print("\n  === CEM planning evaluation (MLP single-step) ===")
    var eval_rng = PhiloxRandom(seed=UInt64(2027), offset=UInt64(0))
    var n_success: Int = 0
    var sum_steps_survived: Int = 0
    var t_eval_start = perf_counter_ns()

    var enc_input_1 = tile[1, ENC_INPUT_DIM](enc_input_buf)
    var enc_hpre_1 = tile[1, ENC_HIDDEN_DIM](enc_hpre_buf)
    var enc_hact_1 = tile[1, ENC_HIDDEN_DIM](enc_hact_buf)
    var enc_output_1 = tile[1, ENC_OUTPUT_DIM](enc_output_buf)

    for ep in range(N_EVAL_EPISODES):
        var u0 = Float64(eval_rng.step_uniform()[0])
        var u1 = Float64(eval_rng.step_uniform()[0])
        var u2 = Float64(eval_rng.step_uniform()[0])
        var u3 = Float64(eval_rng.step_uniform()[0])
        var x = (u0 * 2.0 - 1.0) * CP_RESET_RANGE
        var x_dot = (u1 * 2.0 - 1.0) * CP_RESET_RANGE
        var theta = (u2 * 2.0 - 1.0) * CP_RESET_RANGE
        var theta_dot = (u3 * 2.0 - 1.0) * CP_RESET_RANGE

        unsafe_memset(agent_z_buf, 0, HIDDEN)
        for h in range(PLAN_HORIZON):
            cem_mu[h] = 0.0
            cem_sigma[h] = INITIAL_SIGMA

        for j in range(HIDDEN):
            enc_input_buf[j] = Scalar[dtype](0.0)
        enc_input_buf[HIDDEN] = Scalar[dtype](0.0)
        enc_input_buf[HIDDEN + ACTION_DIM + 0] = Scalar[dtype](x / CP_X_SCALE)
        enc_input_buf[HIDDEN + ACTION_DIM + 1] = Scalar[dtype](
            x_dot / CP_XDOT_SCALE
        )
        enc_input_buf[HIDDEN + ACTION_DIM + 2] = Scalar[dtype](
            theta / CP_THETA_SCALE
        )
        enc_input_buf[HIDDEN + ACTION_DIM + 3] = Scalar[dtype](
            theta_dot / CP_THETADOT_SCALE
        )
        ENC.forward[1, dtype](
            enc_params, enc_input_1, enc_hpre_1, enc_hact_1, enc_output_1
        )
        for j in range(HIDDEN):
            agent_z_buf[j] = enc_output_buf[j]

        var steps_survived: Int = 0
        var terminated_at: Int = -1

        for step in range(MAX_EPISODE_STEPS):
            var x_abs = x if x > 0.0 else -x
            var th_abs = theta if theta > 0.0 else -theta
            if x_abs > CP_X_THRESHOLD or th_abs > CP_THETA_THRESHOLD:
                terminated_at = step
                break
            steps_survived = step + 1

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
                        var x_norm = Float64(cem_mu_obs_buf[s * OBS_DIM + 0])
                        var th_norm = Float64(cem_mu_obs_buf[s * OBS_DIM + 2])
                        var a = Float64(cem_actions_buf[s * PLAN_HORIZON + h])
                        cem_scores[s] -= (
                            th_norm * th_norm
                            + POS_PENALTY * x_norm * x_norm
                            + ACTION_PENALTY * a * a
                        )
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
            var stepped = _step_cartpole(
                x, x_dot, theta, theta_dot, action_norm
            )
            x = stepped[0]
            x_dot = stepped[1]
            theta = stepped[2]
            theta_dot = stepped[3]

            for j in range(HIDDEN):
                enc_input_buf[j] = agent_z_buf[j]
            enc_input_buf[HIDDEN] = Scalar[dtype](action_norm)
            enc_input_buf[HIDDEN + ACTION_DIM + 0] = Scalar[dtype](
                x / CP_X_SCALE
            )
            enc_input_buf[HIDDEN + ACTION_DIM + 1] = Scalar[dtype](
                x_dot / CP_XDOT_SCALE
            )
            enc_input_buf[HIDDEN + ACTION_DIM + 2] = Scalar[dtype](
                theta / CP_THETA_SCALE
            )
            enc_input_buf[HIDDEN + ACTION_DIM + 3] = Scalar[dtype](
                theta_dot / CP_THETADOT_SCALE
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

        sum_steps_survived += steps_survived
        var passed = steps_survived >= PASS_STEPS
        if passed:
            n_success += 1
        if terminated_at == -1:
            print(
                "    ep=",
                ep,
                " : SURVIVED full ",
                MAX_EPISODE_STEPS,
                " steps  →",
                "PASS" if passed else "MISS",
            )
        else:
            print(
                "    ep=",
                ep,
                " : terminated at step ",
                terminated_at,
                " →",
                "PASS" if passed else "MISS",
            )

    var t_eval = Float64(perf_counter_ns() - t_eval_start) / 1e9
    print("\n  eval wall time:", t_eval, "s")
    print("  success rate :", n_success, "/", N_EVAL_EPISODES)
    print(
        "  avg steps survived (all eps):",
        Float64(sum_steps_survived) / Float64(N_EVAL_EPISODES),
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
