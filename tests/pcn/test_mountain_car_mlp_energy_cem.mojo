"""MountainCar Continuous CEM planning — Phase-1 ablation (Experiment 3).

MLP architecture trained with PC-style local energy rules, NO SGLD.

Architecture: identical to PCN baseline (two PCBlocks) since their forward
`μ = tanh(x_below) @ W + b` is the same as a standard MLP layer.

Training (per timestep t in rollout):
1. Encoder forward → z_init = enc(prev_z, action_{t-1}, obs_t). No SGLD.
2. Forward through dynamics (no settling, no inference loop):
     - μ_z   = block_0.predict([prev_z, action])      # transition prediction
     - ε_z   = z_init − μ_z                            # transition residual
     - μ_obs = block_1.predict(z_init)                 # decoder prediction
     - ε_obs = obs_t − μ_obs                           # readout residual
3. Weight gradients via local PC rule (PCBlock.weight_grad bakes the sign):
     - block_0 grads from (ε_z, a_below_0=tanh([prev_z,action]))
     - block_1 grads from (ε_obs, a_below_1=tanh(z_init))
4. Encoder grad = energy gradient w.r.t. z_init (chained through block_1):
     enc_dz = ε_z − act_deriv(z_init) ⊙ pull_back(ε_obs through block_1)
5. Adam steps for block_0+block_1 (PC layout) and encoder.
6. prev_z = z_init (no settling — encoder output is the latent).

Eval: same encoder + dynamics forward, no SGLD anywhere.

Hypothesis (Phase-1 Experiment 3, see docs/PCN_MBRL_PLAN.md):
- If the energy *objective* is the source of PCN's win, this should match PCN.
- If the SGLD-driven settling is the source, this should lose to PCN.

Comparison only — pass criterion (≥ 3/5) reported but not enforced.

Run:
    pixi run mojo run -I . tests/pcn/test_mountain_car_mlp_energy_cem.mojo
"""

from std.math import sqrt, log, cos, sin, tanh, pi
from std.memory import alloc, memset
from std.random.philox import Random as PhiloxRandom
from std.time import perf_counter_ns
from noeira.core.tiles import tile

from noeira.nn.constants import DT as dtype
from noeira.experimental.pcn.pc_initializer import PCXavier
from noeira.experimental.pcn.pc_optimizer import PCAdam
from noeira.experimental.pcn.pc_scheduler import CosineWarmupSchedule
from noeira.experimental.pcn import (
    PCBlock,
    PCEncoder,
    PCSequential,
    PCTanh,
    clip_grad_norm,
)


# MountainCar Continuous physics.
comptime MC_FORCE: Float64 = 0.0015
comptime MC_GRAVITY: Float64 = 0.0025
comptime MC_MAX_SPEED: Float64 = 0.07
comptime MC_MIN_POSITION: Float64 = -1.2
comptime MC_MAX_POSITION: Float64 = 0.6
comptime MC_GOAL_POSITION: Float64 = 0.45
comptime MC_POS_CENTER: Float64 = -0.3
comptime MC_POS_HALF_RANGE: Float64 = 0.9

# Architecture — identical to PCN baseline.
comptime BATCH = 32
comptime HIDDEN = 64
comptime ACTION_DIM = 1
comptime OBS_DIM = 2
comptime AUG_DIM = HIDDEN + ACTION_DIM
comptime SEQ_LEN = 20
comptime EPOCHS = 100
comptime N_BATCHES_PER_EPOCH = 100
comptime WARMUP_EPOCHS = 5
comptime LR_MIN_SCALE: Float64 = 0.1
comptime ADAM_LR_PC: Float64 = 0.001
comptime ADAM_LR_ENC: Float64 = 0.001
comptime GRAD_CLIP_NORM: Float64 = 1.0

comptime ENC_INPUT_DIM = HIDDEN + ACTION_DIM + OBS_DIM
comptime ENC_HIDDEN_DIM = 64
comptime ENC_OUTPUT_DIM = HIDDEN
comptime ENC = PCEncoder[ENC_INPUT_DIM, ENC_HIDDEN_DIM, ENC_OUTPUT_DIM]
comptime ENC_PARAM_SIZE = ENC.PARAM_SIZE

# CEM planning hyperparameters.
comptime PLAN_HORIZON = 20
comptime N_SAMPLES = 128
comptime N_ELITES = 16
comptime N_CEM_ITERS = 2
comptime INITIAL_SIGMA: Float64 = 0.5
comptime MIN_SIGMA: Float64 = 0.05
comptime ACTION_PENALTY: Float64 = 0.001
comptime MAX_EPISODE_STEPS = 200
comptime N_EVAL_EPISODES = 5

# Two PCBlocks (Bogacz canonical: tanh(x) @ W + b).
comptime BLOCK0 = PCBlock[AUG_DIM, HIDDEN, PCTanh]
comptime BLOCK1 = PCBlock[HIDDEN, OBS_DIM, PCTanh]
comptime NET = PCSequential[BLOCK0, BLOCK1]
comptime OPT_PC = PCAdam[LR=ADAM_LR_PC]
comptime OPT_ENC = PCAdam[LR=ADAM_LR_ENC]
comptime SCHED = CosineWarmupSchedule[
    WARMUP_EPOCHS=WARMUP_EPOCHS, MIN_SCALE=LR_MIN_SCALE
]


def _step_mountain_car(
    position: Float64, velocity: Float64, action: Float64
) -> Tuple[Float64, Float64]:
    var u = action
    if u > 1.0:
        u = 1.0
    elif u < -1.0:
        u = -1.0
    var new_v = velocity + u * MC_FORCE - cos(3.0 * position) * MC_GRAVITY
    if new_v > MC_MAX_SPEED:
        new_v = MC_MAX_SPEED
    elif new_v < -MC_MAX_SPEED:
        new_v = -MC_MAX_SPEED
    var new_p = position + new_v
    if new_p < MC_MIN_POSITION:
        new_p = MC_MIN_POSITION
        new_v = 0.0
    elif new_p > MC_MAX_POSITION:
        new_p = MC_MAX_POSITION
        new_v = 0.0
    return (new_p, new_v)


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
    var position = -0.6 + u0 * 0.2
    var velocity = 0.0
    obs_buf[obs_offset + 0] = Scalar[dtype](
        (position - MC_POS_CENTER) / MC_POS_HALF_RANGE
    )
    obs_buf[obs_offset + 1] = Scalar[dtype](velocity / MC_MAX_SPEED)
    for t in range(SEQ_LEN_T):
        var ua = Float64(rng.step_uniform()[0])
        var action_norm = ua * 2.0 - 1.0
        actions_buf[actions_offset + t] = Scalar[dtype](action_norm)
        var stepped = _step_mountain_car(position, velocity, action_norm)
        position = stepped[0]
        velocity = stepped[1]
        obs_buf[obs_offset + (t + 1) * OBS_DIM + 0] = Scalar[dtype](
            (position - MC_POS_CENTER) / MC_POS_HALF_RANGE
        )
        obs_buf[obs_offset + (t + 1) * OBS_DIM + 1] = Scalar[dtype](
            velocity / MC_MAX_SPEED
        )


def _gauss_pair(mut rng: PhiloxRandom) -> Tuple[Float64, Float64]:
    var u1 = Float64(rng.step_uniform()[0])
    var u2 = Float64(rng.step_uniform()[0])
    if u1 < 1e-12:
        u1 = 1e-12
    var r = sqrt(-2.0 * log(u1))
    var theta = 2.0 * pi * u2
    return (r * cos(theta), r * sin(theta))


def main() raises:
    print("=" * 60)
    print("MountainCar Continuous — Exp 3: MLP-arch + energy local rules")
    print("=" * 60)
    print("  Arch       : same as PCN baseline (PCBlock × 2, PCTanh)")
    print("  PC params  :", NET.PARAM_SIZE)
    print(
        "  Enc arch   : MLP[",
        ENC_INPUT_DIM,
        "→",
        ENC_HIDDEN_DIM,
        "→",
        ENC_OUTPUT_DIM,
        "]",
    )
    print(
        "  Training   : NO SGLD; per-step PC weight rule + energy ∂E/∂z to"
        " encoder"
    )
    print("  Eval       : encoder forward + MLP imagination, NO SGLD")

    # ── PC params + Adam state (PCSequential layout) ─────────────────────────
    var pc_params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var pc_grads_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var pc_opt_state_buf = alloc[Scalar[dtype]](
        NET.PARAM_SIZE * OPT_PC.STATE_PER_PARAM
    ).as_unsafe_any_origin()
    var pc_opt_global_buf = alloc[Scalar[dtype]](OPT_PC.GLOBAL_STATE_SIZE).as_unsafe_any_origin()
    memset(pc_params_buf, 0, NET.PARAM_SIZE)
    memset(pc_grads_buf, 0, NET.PARAM_SIZE)
    memset(pc_opt_state_buf, 0, NET.PARAM_SIZE * OPT_PC.STATE_PER_PARAM)
    memset(pc_opt_global_buf, 0, OPT_PC.GLOBAL_STATE_SIZE)
    var pc_params = tile[NET.PARAM_SIZE](pc_params_buf)
    var pc_grads = tile[NET.PARAM_SIZE](pc_grads_buf)
    var pc_opt_state = tile[NET.PARAM_SIZE, OPT_PC.STATE_PER_PARAM](pc_opt_state_buf)
    var pc_opt_global = tile[OPT_PC.GLOBAL_STATE_SIZE](pc_opt_global_buf)
    NET.pc_init_params[PCXavier, dtype](pc_params)

    # Per-block param + grad views (used by PCBlock primitives directly).
    comptime offset_b1 = NET._param_offset[1]()
    var params_b0 = tile[BLOCK0.PARAM_SIZE](pc_params_buf)
    var params_b1 = tile[BLOCK1.PARAM_SIZE](pc_params_buf + offset_b1)
    var grads_b0 = tile[BLOCK0.PARAM_SIZE](pc_grads_buf)
    var grads_b1 = tile[BLOCK1.PARAM_SIZE](pc_grads_buf + offset_b1)

    # ── Encoder params + Adam state ──────────────────────────────────────────
    var enc_params_buf = alloc[Scalar[dtype]](ENC_PARAM_SIZE).as_unsafe_any_origin()
    var enc_grads_buf = alloc[Scalar[dtype]](ENC_PARAM_SIZE).as_unsafe_any_origin()
    var enc_opt_state_buf = alloc[Scalar[dtype]](
        ENC_PARAM_SIZE * OPT_ENC.STATE_PER_PARAM
    ).as_unsafe_any_origin()
    var enc_opt_global_buf = alloc[Scalar[dtype]](OPT_ENC.GLOBAL_STATE_SIZE).as_unsafe_any_origin()
    memset(enc_params_buf, 0, ENC_PARAM_SIZE)
    memset(enc_grads_buf, 0, ENC_PARAM_SIZE)
    memset(enc_opt_state_buf, 0, ENC_PARAM_SIZE * OPT_ENC.STATE_PER_PARAM)
    memset(enc_opt_global_buf, 0, OPT_ENC.GLOBAL_STATE_SIZE)
    var enc_params = tile[ENC_PARAM_SIZE](enc_params_buf)
    var enc_grads = tile[ENC_PARAM_SIZE](enc_grads_buf)
    var enc_opt_state = tile[ENC_PARAM_SIZE, OPT_ENC.STATE_PER_PARAM](enc_opt_state_buf)
    var enc_opt_global = tile[OPT_ENC.GLOBAL_STATE_SIZE](enc_opt_global_buf)
    ENC.xavier_init[dtype](enc_params, UInt64(123))

    # ── Per-block scratch (BATCH=32 training, BATCH=1 views for eval) ─────────
    # BLOCK0 forward: x_aug_buf[BATCH, AUG_DIM] → mu_z_buf[BATCH, HIDDEN]; cache a_aug[BATCH, AUG_DIM].
    # BLOCK1 forward: z_buf[BATCH, HIDDEN] → mu_obs_buf[BATCH, OBS_DIM]; cache a_z[BATCH, HIDDEN].
    var x_aug_buf = alloc[Scalar[dtype]](BATCH * AUG_DIM).as_unsafe_any_origin()
    var a_aug_buf = alloc[Scalar[dtype]](BATCH * AUG_DIM).as_unsafe_any_origin()
    var mu_z_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var eps_z_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var z_init_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var a_z_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var mu_obs_buf = alloc[Scalar[dtype]](BATCH * OBS_DIM).as_unsafe_any_origin()
    var eps_obs_buf = alloc[Scalar[dtype]](BATCH * OBS_DIM).as_unsafe_any_origin()
    var y_obs_buf = alloc[Scalar[dtype]](BATCH * OBS_DIM).as_unsafe_any_origin()
    var pull_back_buf = alloc[Scalar[dtype]](
        BATCH * HIDDEN
    ).as_unsafe_any_origin()  # block_1.pull_back output (BATCH, HIDDEN)
    var gated_buf = alloc[Scalar[dtype]](
        BATCH * HIDDEN
    ).as_unsafe_any_origin()  # act_deriv ⊙ pull_back

    var x_aug = tile[BATCH, AUG_DIM](x_aug_buf)
    var a_aug = tile[BATCH, AUG_DIM](a_aug_buf)
    var mu_z = tile[BATCH, HIDDEN](mu_z_buf)
    var z_init = tile[BATCH, HIDDEN](z_init_buf)
    var a_z = tile[BATCH, HIDDEN](a_z_buf)
    var mu_obs = tile[BATCH, OBS_DIM](mu_obs_buf)
    var y_obs = tile[BATCH, OBS_DIM](y_obs_buf)
    var eps_z = tile[BATCH, HIDDEN](eps_z_buf)
    var eps_obs = tile[BATCH, OBS_DIM](eps_obs_buf)
    var pull_back_out = tile[BATCH, HIDDEN](pull_back_buf)
    var gated = tile[BATCH, HIDDEN](gated_buf)

    # ── Encoder scratch ──────────────────────────────────────────────────────
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

    var enc_input_1 = tile[1, ENC_INPUT_DIM](enc_input_buf)
    var enc_hpre_1 = tile[1, ENC_HIDDEN_DIM](enc_hpre_buf)
    var enc_hact_1 = tile[1, ENC_HIDDEN_DIM](enc_hact_buf)
    var enc_output_1 = tile[1, ENC_OUTPUT_DIM](enc_output_buf)

    var actions_buf = alloc[Scalar[dtype]](BATCH * SEQ_LEN).as_unsafe_any_origin()
    var obs_buf = alloc[Scalar[dtype]](BATCH * (SEQ_LEN + 1) * OBS_DIM).as_unsafe_any_origin()
    var prev_z_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()

    # ── Train ────────────────────────────────────────────────────────────────
    print("\n  epoch | last_obs_loss | wall_t (s)")
    print("  ------+---------------+------------")
    var rng = PhiloxRandom(seed=UInt64(7), offset=UInt64(0))
    var pc_step_num: Int = 0
    var enc_step_num: Int = 0
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

            for t in range(0, SEQ_LEN + 1):
                # 1. Build encoder input + x_aug + y_obs.
                for b in range(BATCH):
                    for j in range(HIDDEN):
                        enc_input_buf[b * ENC_INPUT_DIM + j] = prev_z_buf[
                            b * HIDDEN + j
                        ]
                    var act_val = (
                        Scalar[dtype](0.0) if t
                        == 0 else actions_buf[b * SEQ_LEN + (t - 1)]
                    )
                    enc_input_buf[b * ENC_INPUT_DIM + HIDDEN] = act_val
                    for d in range(OBS_DIM):
                        enc_input_buf[
                            b * ENC_INPUT_DIM + HIDDEN + ACTION_DIM + d
                        ] = obs_buf[
                            b * (SEQ_LEN + 1) * OBS_DIM + t * OBS_DIM + d
                        ]
                    for j in range(HIDDEN):
                        x_aug_buf[b * AUG_DIM + j] = prev_z_buf[b * HIDDEN + j]
                    x_aug_buf[b * AUG_DIM + HIDDEN] = act_val
                    for d in range(OBS_DIM):
                        y_obs_buf[b * OBS_DIM + d] = obs_buf[
                            b * (SEQ_LEN + 1) * OBS_DIM + t * OBS_DIM + d
                        ]

                # 2. Encoder forward → z_init.
                ENC.forward[BATCH, dtype](
                    enc_params, enc_input, enc_hpre, enc_hact, enc_output
                )
                for b in range(BATCH):
                    for j in range(HIDDEN):
                        z_init_buf[b * HIDDEN + j] = enc_output_buf[
                            b * ENC_OUTPUT_DIM + j
                        ]

                # 3. block_0.predict([prev_z, action]) → mu_z; cache a_aug.
                BLOCK0.predict[BATCH, dtype](x_aug, params_b0, mu_z, a_aug)
                # ε_z = z_init − μ_z.
                BLOCK0.eps_compute[BATCH, dtype](z_init, mu_z, eps_z)

                # 4. block_1.predict(z_init) → mu_obs; cache a_z = tanh(z_init).
                BLOCK1.predict[BATCH, dtype](z_init, params_b1, mu_obs, a_z)
                # ε_obs = obs_t − μ_obs.
                BLOCK1.eps_compute[BATCH, dtype](y_obs, mu_obs, eps_obs)

                # 5. Track output loss.
                var sum_sq: Float64 = 0.0
                for b in range(BATCH):
                    for d in range(OBS_DIM):
                        var e = Float64(eps_obs_buf[b * OBS_DIM + d])
                        sum_sq += e * e
                last_loss = 0.5 * sum_sq / Float64(BATCH)

                # 6. Local PC weight grads (sign baked in).
                memset(pc_grads_buf, 0, NET.PARAM_SIZE)
                BLOCK0.weight_grad[BATCH, dtype](eps_z, a_aug, grads_b0)
                BLOCK1.weight_grad[BATCH, dtype](eps_obs, a_z, grads_b1)

                # 7. Encoder grad = ε_z − act_deriv(z_init) ⊙ pull_back(ε_obs).
                BLOCK1.pull_back[BATCH, dtype](
                    eps_obs, params_b1, pull_back_out
                )
                # gated[b, j] = pull_back[b, j] * (1 − tanh²(z_init[b, j])).
                BLOCK1.act_derivative_mul[BATCH, dtype](
                    z_init, pull_back_out, gated
                )
                for b in range(BATCH):
                    for j in range(HIDDEN):
                        enc_dz_buf[b * ENC_OUTPUT_DIM + j] = Scalar[dtype](
                            Float64(eps_z_buf[b * HIDDEN + j])
                            - Float64(gated_buf[b * HIDDEN + j])
                        )
                ENC.backward[BATCH, dtype](
                    enc_params, enc_input, enc_hact, enc_dz, enc_grads
                )

                # 8. Clip + Adam.
                clip_grad_norm[NET.PARAM_SIZE, dtype](pc_grads, GRAD_CLIP_NORM)
                pc_step_num += 1
                OPT_PC.step[NET.PARAM_SIZE, dtype](
                    pc_params,
                    pc_grads,
                    pc_opt_state,
                    pc_opt_global,
                    pc_step_num,
                    lr_scale=lr_scale,
                )
                clip_grad_norm[ENC_PARAM_SIZE, dtype](enc_grads, GRAD_CLIP_NORM)
                enc_step_num += 1
                OPT_ENC.step[ENC_PARAM_SIZE, dtype](
                    enc_params,
                    enc_grads,
                    enc_opt_state,
                    enc_opt_global,
                    enc_step_num,
                    lr_scale=lr_scale,
                )

                # 9. prev_z = z_init (no settling).
                for b in range(BATCH):
                    for j in range(HIDDEN):
                        prev_z_buf[b * HIDDEN + j] = z_init_buf[b * HIDDEN + j]

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

    # ────────────────────────────────────────────────────────────────────────
    # CEM eval (encoder + dynamics forward, no SGLD).
    # ────────────────────────────────────────────────────────────────────────
    var cem_z_buf = alloc[Scalar[dtype]](N_SAMPLES * HIDDEN).as_unsafe_any_origin()
    var cem_z_next_buf = alloc[Scalar[dtype]](N_SAMPLES * HIDDEN).as_unsafe_any_origin()
    var cem_x_in_buf = alloc[Scalar[dtype]](N_SAMPLES * AUG_DIM).as_unsafe_any_origin()
    var cem_a_z_buf = alloc[Scalar[dtype]](N_SAMPLES * AUG_DIM).as_unsafe_any_origin()
    var cem_a_s_buf = alloc[Scalar[dtype]](N_SAMPLES * HIDDEN).as_unsafe_any_origin()
    var cem_obs_pred_buf = alloc[Scalar[dtype]](N_SAMPLES * OBS_DIM).as_unsafe_any_origin()
    var cem_actions_buf = alloc[Scalar[dtype]](N_SAMPLES * PLAN_HORIZON).as_unsafe_any_origin()
    var cem_x_in = tile[N_SAMPLES, AUG_DIM](cem_x_in_buf)
    var cem_a_z = tile[N_SAMPLES, AUG_DIM](cem_a_z_buf)
    var cem_z_next = tile[N_SAMPLES, HIDDEN](cem_z_next_buf)
    var cem_a_s = tile[N_SAMPLES, HIDDEN](cem_a_s_buf)
    var cem_obs_pred = tile[N_SAMPLES, OBS_DIM](cem_obs_pred_buf)

    var cem_mu = List[Float64](capacity=PLAN_HORIZON)
    var cem_sigma = List[Float64](capacity=PLAN_HORIZON)
    var cem_scores = List[Float64](capacity=N_SAMPLES)
    var cem_max_pos = List[Float64](capacity=N_SAMPLES)
    var cem_indices = List[Int](capacity=N_SAMPLES)
    for _ in range(PLAN_HORIZON):
        cem_mu.append(0.0)
        cem_sigma.append(INITIAL_SIGMA)
    for _ in range(N_SAMPLES):
        cem_scores.append(0.0)
        cem_max_pos.append(-2.0)
        cem_indices.append(0)

    var agent_z_buf = alloc[Scalar[dtype]](HIDDEN).as_unsafe_any_origin()

    print("\n  === CEM planning evaluation (no SGLD anywhere) ===")
    var eval_rng = PhiloxRandom(seed=UInt64(2027), offset=UInt64(0))
    var n_success: Int = 0
    var sum_steps_to_goal: Int = 0
    var t_eval_start = perf_counter_ns()

    for ep in range(N_EVAL_EPISODES):
        var u0 = Float64(eval_rng.step_uniform()[0])
        var position = -0.6 + u0 * 0.2
        var velocity = 0.0
        var max_position_seen: Float64 = position
        var reached_goal = False
        var step_at_goal: Int = -1

        memset(agent_z_buf, 0, HIDDEN)
        for h in range(PLAN_HORIZON):
            cem_mu[h] = 0.0
            cem_sigma[h] = INITIAL_SIGMA

        # Bootstrap encode (no SGLD).
        for j in range(HIDDEN):
            enc_input_buf[j] = Scalar[dtype](0.0)
        enc_input_buf[HIDDEN] = Scalar[dtype](0.0)
        enc_input_buf[HIDDEN + ACTION_DIM + 0] = Scalar[dtype](
            (position - MC_POS_CENTER) / MC_POS_HALF_RANGE
        )
        enc_input_buf[HIDDEN + ACTION_DIM + 1] = Scalar[dtype](
            velocity / MC_MAX_SPEED
        )
        ENC.forward[1, dtype](
            enc_params, enc_input_1, enc_hpre_1, enc_hact_1, enc_output_1
        )
        for j in range(HIDDEN):
            agent_z_buf[j] = enc_output_buf[j]

        for step in range(MAX_EPISODE_STEPS):
            if position >= MC_GOAL_POSITION and not reached_goal:
                reached_goal = True
                step_at_goal = step
                n_success += 1
                sum_steps_to_goal += step
                break

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
                    cem_max_pos[s] = -2.0
                    cem_scores[s] = 0.0

                for h in range(PLAN_HORIZON):
                    for s in range(N_SAMPLES):
                        for j in range(HIDDEN):
                            cem_x_in_buf[s * AUG_DIM + j] = cem_z_buf[
                                s * HIDDEN + j
                            ]
                        cem_x_in_buf[s * AUG_DIM + HIDDEN] = cem_actions_buf[
                            s * PLAN_HORIZON + h
                        ]
                    BLOCK0.predict[N_SAMPLES, dtype](
                        cem_x_in, params_b0, cem_z_next, cem_a_z
                    )
                    BLOCK1.predict[N_SAMPLES, dtype](
                        cem_z_next, params_b1, cem_obs_pred, cem_a_s
                    )
                    for s in range(N_SAMPLES):
                        var pos_norm = Float64(
                            cem_obs_pred_buf[s * OBS_DIM + 0]
                        )
                        var pos = pos_norm * MC_POS_HALF_RANGE + MC_POS_CENTER
                        if pos > cem_max_pos[s]:
                            cem_max_pos[s] = pos
                        var a = Float64(cem_actions_buf[s * PLAN_HORIZON + h])
                        cem_scores[s] -= ACTION_PENALTY * a * a
                    for s in range(N_SAMPLES):
                        for j in range(HIDDEN):
                            cem_z_buf[s * HIDDEN + j] = cem_z_next_buf[
                                s * HIDDEN + j
                            ]

                for s in range(N_SAMPLES):
                    cem_scores[s] += cem_max_pos[s]

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

            var action = cem_mu[0]
            if action > 1.0:
                action = 1.0
            elif action < -1.0:
                action = -1.0
            var stepped = _step_mountain_car(position, velocity, action)
            position = stepped[0]
            velocity = stepped[1]
            if position > max_position_seen:
                max_position_seen = position

            for j in range(HIDDEN):
                enc_input_buf[j] = agent_z_buf[j]
            enc_input_buf[HIDDEN] = Scalar[dtype](action)
            enc_input_buf[HIDDEN + ACTION_DIM + 0] = Scalar[dtype](
                (position - MC_POS_CENTER) / MC_POS_HALF_RANGE
            )
            enc_input_buf[HIDDEN + ACTION_DIM + 1] = Scalar[dtype](
                velocity / MC_MAX_SPEED
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

        if not reached_goal:
            print(
                "    ep=",
                ep,
                " : MISS (max_position=",
                max_position_seen,
                ")",
            )
        else:
            print(
                "    ep=",
                ep,
                " : GOAL at step",
                step_at_goal,
                " (max_position=",
                max_position_seen,
                ")",
            )

    var t_eval = Float64(perf_counter_ns() - t_eval_start) / 1e9
    print("\n  eval wall time:", t_eval, "s")
    print("  success rate :", n_success, "/", N_EVAL_EPISODES)
    if n_success > 0:
        print(
            "  avg steps to goal (successful eps):",
            Float64(sum_steps_to_goal) / Float64(n_success),
        )

    print("\n  === Exp 3 (MLP + energy local rules, no SGLD) summary ===")
    print(
        "  Solved",
        n_success,
        "/",
        N_EVAL_EPISODES,
        " (PCN baseline: 5/5; MLP+BPTT: 3/5; MLP-1step: 0/5)",
    )

    # cleanup
    pc_params_buf.free()
    pc_grads_buf.free()
    pc_opt_state_buf.free()
    pc_opt_global_buf.free()
    enc_params_buf.free()
    enc_grads_buf.free()
    enc_opt_state_buf.free()
    enc_opt_global_buf.free()
    x_aug_buf.free()
    a_aug_buf.free()
    mu_z_buf.free()
    eps_z_buf.free()
    z_init_buf.free()
    a_z_buf.free()
    mu_obs_buf.free()
    eps_obs_buf.free()
    y_obs_buf.free()
    pull_back_buf.free()
    gated_buf.free()
    enc_input_buf.free()
    enc_hpre_buf.free()
    enc_hact_buf.free()
    enc_output_buf.free()
    enc_dz_buf.free()
    actions_buf.free()
    obs_buf.free()
    prev_z_buf.free()
    cem_z_buf.free()
    cem_z_next_buf.free()
    cem_x_in_buf.free()
    cem_a_z_buf.free()
    cem_a_s_buf.free()
    cem_obs_pred_buf.free()
    cem_actions_buf.free()
    agent_z_buf.free()
    print("=== Done ===")
