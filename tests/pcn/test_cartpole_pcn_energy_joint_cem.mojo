"""CartPole-Continuous — joint training (PCN dynamics + value head).

Extension of Phase-1 Experiment 3 (PC weight rule on MLP dynamics, no SGLD)
that tests whether the PC bias on the dynamics weights survives value-loss
gradient pressure during joint training.

Architecture additions over Exp 3:
  - Value head:  V(μ_z) = W_val @ μ_z + b_val           (Linear[HIDDEN, 1])
  - Value target: MC discounted return per (b, t) on the training rollouts
    (γ=0.99, rollouts terminate on |x|>2.4 or |θ|>0.2, post-term reward = 0)

Per-timestep training (no BPTT — same as Exp 3):
  z_init = enc(prev_z, action_{t-1}, obs_t)
  μ_z    = block_0.predict([prev_z, action])
  μ_obs  = block_1.predict(z_init)
  ε_z    = z_init − μ_z
  ε_obs  = obs_t − μ_obs
  V_pred = W_val @ μ_z + b_val
  V_tgt  = MC return from t (γ=0.99, with termination)
  dV     = V_pred − V_tgt

Gradient routing (the central design decision):
  block_0:   weight_grad(ε_z, a_aug)          ← PC weight rule       (PC contribution)
           + LAMBDA_VAL · weight_grad(ε_val, a_aug)                    (value contribution)
             where ε_val[b,j] = −dV[b] · W_val[j]   (so block_0.weight_grad
             returns +dL_val/dW_dyn — sign convention matches the PC rule)
  block_1:   weight_grad(ε_obs, a_z)          ← decoder reconstruction
  enc    :   ε_z − act_deriv(z_init) ⊙ pull_back(ε_obs)               (unchanged)
  V head :   dL_val/dW_val[j] = Σ_b dV[b] · μ_z[b,j]                   (standard SGD)
             dL_val/db_val     = Σ_b dV[b]
  prev_z stop-gradient at the timestep boundary (no BPTT across t)

The dynamics weights now receive BOTH the PC rule's "z_init ↔ μ_z" pressure
AND the value head's "μ_z must support value prediction" pressure. If the
PC bias survives joint training, dynamics produces a rollable AND
value-aware latent, and CEM planning still solves 5/5. If the value
gradient washes out the PC bias, dynamics drifts toward MLP+BPTT-equivalent
and CEM eval may degrade.

Eval: identical CEM planning to Exp 3 — encoder + dynamics forward, no
value head used at planning time (the value head exists only as
joint-training pressure on the dynamics).

See `docs/PCN_MBRL_DESIGN.md` Phase-1 joint-training experiment.

Run:
    pixi run mojo run -I . tests/pcn/test_cartpole_pcn_energy_joint_cem.mojo
"""

from std.math import sqrt, log, cos, sin, tanh, pi
from std.memory import alloc, unsafe_memset
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


# CartPole-Continuous physics.
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

# Architecture — identical to PCN baseline.
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
comptime ADAM_LR_PC: Float64 = 0.001
comptime ADAM_LR_ENC: Float64 = 0.001
comptime GRAD_CLIP_NORM: Float64 = 1.0

comptime ENC_INPUT_DIM = HIDDEN + ACTION_DIM + OBS_DIM
comptime ENC_HIDDEN_DIM = 64
comptime ENC_OUTPUT_DIM = HIDDEN
comptime ENC = PCEncoder[ENC_INPUT_DIM, ENC_HIDDEN_DIM, ENC_OUTPUT_DIM]
comptime ENC_PARAM_SIZE = ENC.PARAM_SIZE

# ── Joint-training (value head) constants ────────────────────────────────────
# V(μ_z) = W_val[HIDDEN] · μ_z + b_val   (scalar value prediction)
# Value head trained jointly with the dynamics; value-loss gradient flows
# back into BLOCK0 (dynamics) weights alongside the PC rule.
comptime V_PARAM_SIZE = HIDDEN + 1  # W_val (HIDDEN) | b_val (1)
comptime V_W_OFFSET = 0
comptime V_B_OFFSET = HIDDEN
comptime GAMMA: Float64 = 0.99
# Max MC return for SEQ_LEN+1 = 21 steps survived = (1 − γ^21) / (1 − γ) ≈ 19.0
# Normalize value targets by V_MAX so V_pred and dV stay O(1) at init.
comptime V_MAX: Float64 = (1.0 - 0.99**21) / (1.0 - 0.99)
# Weight on value loss relative to PC energy. 1.0 = equal weighting.
# 0.0 = pure PC (sanity check: should reproduce Exp-3 5/5 result).
comptime LAMBDA_VAL: Float64 = 0.0
# Value head uses its own Adam (same hyperparams as encoder/dynamics).
comptime OPT_VAL = PCAdam[LR=ADAM_LR_PC]

# CEM planning hyperparameters.
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

comptime BLOCK0 = PCBlock[AUG_DIM, HIDDEN, PCTanh]
comptime BLOCK1 = PCBlock[HIDDEN, OBS_DIM, PCTanh]
comptime NET = PCSequential[BLOCK0, BLOCK1]
comptime OPT_PC = PCAdam[LR=ADAM_LR_PC]
comptime OPT_ENC = PCAdam[LR=ADAM_LR_ENC]
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
    var new_x = x + CP_TAU * x_dot
    var new_x_dot = x_dot + CP_TAU * x_acc
    var new_theta = theta + CP_TAU * theta_dot
    var new_theta_dot = theta_dot + CP_TAU * theta_acc
    return (new_x, new_x_dot, new_theta, new_theta_dot)


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


def main() raises:
    print("=" * 60)
    print("CartPole-Continuous — joint training (PCN dynamics + value head)")
    print("=" * 60)
    print(
        "  Arch       : same as PCN baseline (PCBlock × 2, PCTanh) + value head"
    )
    print("  PC params  :", NET.PARAM_SIZE)
    print("  V  params  :", V_PARAM_SIZE, " (HIDDEN+1)")
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
        "  Training   : PC weight rule on dynamics + value loss (λ=",
        LAMBDA_VAL,
        ")",
    )
    print(
        "  V target   : MC discounted return (γ=", GAMMA, ", V_MAX=", V_MAX, ")"
    )
    print(
        "  Eval       : encoder + MLP imagination + CEM (value head unused at"
        " eval)"
    )
    print(
        "  Pass       : survive ≥",
        PASS_STEPS,
        " of",
        MAX_EPISODE_STEPS,
        " steps",
    )

    # ── PC params + Adam state ────────────────────────────────────────────────
    var pc_params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE)
    var pc_grads_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE)
    var pc_opt_state_buf = alloc[Scalar[dtype]](
        NET.PARAM_SIZE * OPT_PC.STATE_PER_PARAM
    )
    var pc_opt_global_buf = alloc[Scalar[dtype]](OPT_PC.GLOBAL_STATE_SIZE)
    unsafe_memset(pc_params_buf, 0, NET.PARAM_SIZE)
    unsafe_memset(pc_grads_buf, 0, NET.PARAM_SIZE)
    unsafe_memset(pc_opt_state_buf, 0, NET.PARAM_SIZE * OPT_PC.STATE_PER_PARAM)
    unsafe_memset(pc_opt_global_buf, 0, OPT_PC.GLOBAL_STATE_SIZE)
    var pc_params = tile[NET.PARAM_SIZE](pc_params_buf.as_unsafe_any_origin())
    var pc_grads = tile[NET.PARAM_SIZE](pc_grads_buf.as_unsafe_any_origin())
    var pc_opt_state = tile[NET.PARAM_SIZE, OPT_PC.STATE_PER_PARAM](pc_opt_state_buf.as_unsafe_any_origin())
    var pc_opt_global = tile[OPT_PC.GLOBAL_STATE_SIZE](pc_opt_global_buf.as_unsafe_any_origin())
    NET.pc_init_params[PCXavier, dtype](pc_params)

    comptime offset_b1 = NET._param_offset[1]()
    var params_b0 = tile[BLOCK0.PARAM_SIZE](pc_params_buf.as_unsafe_any_origin())
    var params_b1 = tile[BLOCK1.PARAM_SIZE](pc_params_buf.as_unsafe_any_origin() + offset_b1)
    var grads_b0 = tile[BLOCK0.PARAM_SIZE](pc_grads_buf.as_unsafe_any_origin())
    var grads_b1 = tile[BLOCK1.PARAM_SIZE](pc_grads_buf.as_unsafe_any_origin() + offset_b1)

    # ── Encoder params + Adam state ──────────────────────────────────────────
    var enc_params_buf = alloc[Scalar[dtype]](ENC_PARAM_SIZE)
    var enc_grads_buf = alloc[Scalar[dtype]](ENC_PARAM_SIZE)
    var enc_opt_state_buf = alloc[Scalar[dtype]](
        ENC_PARAM_SIZE * OPT_ENC.STATE_PER_PARAM
    )
    var enc_opt_global_buf = alloc[Scalar[dtype]](OPT_ENC.GLOBAL_STATE_SIZE)
    unsafe_memset(enc_params_buf, 0, ENC_PARAM_SIZE)
    unsafe_memset(enc_grads_buf, 0, ENC_PARAM_SIZE)
    unsafe_memset(enc_opt_state_buf, 0, ENC_PARAM_SIZE * OPT_ENC.STATE_PER_PARAM)
    unsafe_memset(enc_opt_global_buf, 0, OPT_ENC.GLOBAL_STATE_SIZE)
    var enc_params = tile[ENC_PARAM_SIZE](enc_params_buf.as_unsafe_any_origin())
    var enc_grads = tile[ENC_PARAM_SIZE](enc_grads_buf.as_unsafe_any_origin())
    var enc_opt_state = tile[ENC_PARAM_SIZE, OPT_ENC.STATE_PER_PARAM](enc_opt_state_buf.as_unsafe_any_origin())
    var enc_opt_global = tile[OPT_ENC.GLOBAL_STATE_SIZE](enc_opt_global_buf.as_unsafe_any_origin())
    ENC.xavier_init[dtype](enc_params, UInt64(123))

    # ── Value head params + Adam state ───────────────────────────────────────
    var v_params_buf = alloc[Scalar[dtype]](V_PARAM_SIZE)
    var v_grads_buf = alloc[Scalar[dtype]](V_PARAM_SIZE)
    var v_opt_state_buf = alloc[Scalar[dtype]](
        V_PARAM_SIZE * OPT_VAL.STATE_PER_PARAM
    )
    var v_opt_global_buf = alloc[Scalar[dtype]](OPT_VAL.GLOBAL_STATE_SIZE)
    unsafe_memset(v_params_buf, 0, V_PARAM_SIZE)
    unsafe_memset(v_grads_buf, 0, V_PARAM_SIZE)
    unsafe_memset(v_opt_state_buf, 0, V_PARAM_SIZE * OPT_VAL.STATE_PER_PARAM)
    unsafe_memset(v_opt_global_buf, 0, OPT_VAL.GLOBAL_STATE_SIZE)
    # Xavier-uniform init on W_val: bound = sqrt(6 / (HIDDEN + 1)).
    var v_rng = PhiloxRandom(seed=UInt64(31), offset=UInt64(0))
    var v_bound = sqrt(Float64(6.0) / Float64(HIDDEN + 1))
    for i in range(HIDDEN):
        var u = Float64(v_rng.step_uniform()[0])
        v_params_buf[V_W_OFFSET + i] = Scalar[dtype]((u * 2.0 - 1.0) * v_bound)
    # b_val starts at 0.
    v_params_buf[V_B_OFFSET] = Scalar[dtype](0.0)
    var v_params = tile[V_PARAM_SIZE](v_params_buf.as_unsafe_any_origin())
    var v_grads = tile[V_PARAM_SIZE](v_grads_buf.as_unsafe_any_origin())
    var v_opt_state = tile[V_PARAM_SIZE, OPT_VAL.STATE_PER_PARAM](v_opt_state_buf.as_unsafe_any_origin())
    var v_opt_global = tile[OPT_VAL.GLOBAL_STATE_SIZE](v_opt_global_buf.as_unsafe_any_origin())

    # Buffer for the value-loss contribution to block_0 dynamics grads.
    # weight_grad signature requires a LayoutTensor; allocate parallel to grads_b0.
    var grads_b0_val_buf = alloc[Scalar[dtype]](BLOCK0.PARAM_SIZE)
    var grads_b0_val = tile[BLOCK0.PARAM_SIZE](grads_b0_val_buf.as_unsafe_any_origin())
    # Per-step value scratch.
    var eps_val_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)
    var eps_val = tile[BATCH, HIDDEN](eps_val_buf.as_unsafe_any_origin())
    var t_term_buf = alloc[Scalar[dtype]](
        BATCH
    )  # termination step per rollout, as float

    # ── Per-step scratch ──────────────────────────────────────────────────────
    var x_aug_buf = alloc[Scalar[dtype]](BATCH * AUG_DIM)
    var a_aug_buf = alloc[Scalar[dtype]](BATCH * AUG_DIM)
    var mu_z_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)
    var eps_z_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)
    var z_init_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)
    var a_z_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)
    var mu_obs_buf = alloc[Scalar[dtype]](BATCH * OBS_DIM)
    var eps_obs_buf = alloc[Scalar[dtype]](BATCH * OBS_DIM)
    var y_obs_buf = alloc[Scalar[dtype]](BATCH * OBS_DIM)
    var pull_back_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)
    var gated_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)

    var x_aug = tile[BATCH, AUG_DIM](x_aug_buf.as_unsafe_any_origin())
    var a_aug = tile[BATCH, AUG_DIM](a_aug_buf.as_unsafe_any_origin())
    var mu_z = tile[BATCH, HIDDEN](mu_z_buf.as_unsafe_any_origin())
    var z_init = tile[BATCH, HIDDEN](z_init_buf.as_unsafe_any_origin())
    var a_z = tile[BATCH, HIDDEN](a_z_buf.as_unsafe_any_origin())
    var mu_obs = tile[BATCH, OBS_DIM](mu_obs_buf.as_unsafe_any_origin())
    var y_obs = tile[BATCH, OBS_DIM](y_obs_buf.as_unsafe_any_origin())
    var eps_z = tile[BATCH, HIDDEN](eps_z_buf.as_unsafe_any_origin())
    var eps_obs = tile[BATCH, OBS_DIM](eps_obs_buf.as_unsafe_any_origin())
    var pull_back_out = tile[BATCH, HIDDEN](pull_back_buf.as_unsafe_any_origin())
    var gated = tile[BATCH, HIDDEN](gated_buf.as_unsafe_any_origin())

    var enc_input_buf = alloc[Scalar[dtype]](BATCH * ENC_INPUT_DIM)
    var enc_hpre_buf = alloc[Scalar[dtype]](BATCH * ENC_HIDDEN_DIM)
    var enc_hact_buf = alloc[Scalar[dtype]](BATCH * ENC_HIDDEN_DIM)
    var enc_output_buf = alloc[Scalar[dtype]](BATCH * ENC_OUTPUT_DIM)
    var enc_dz_buf = alloc[Scalar[dtype]](BATCH * ENC_OUTPUT_DIM)
    var enc_input = tile[BATCH, ENC_INPUT_DIM](enc_input_buf.as_unsafe_any_origin())
    var enc_hpre = tile[BATCH, ENC_HIDDEN_DIM](enc_hpre_buf.as_unsafe_any_origin())
    var enc_hact = tile[BATCH, ENC_HIDDEN_DIM](enc_hact_buf.as_unsafe_any_origin())
    var enc_output = tile[BATCH, ENC_OUTPUT_DIM](enc_output_buf.as_unsafe_any_origin())
    var enc_dz = tile[BATCH, ENC_OUTPUT_DIM](enc_dz_buf.as_unsafe_any_origin())

    var enc_input_1 = tile[1, ENC_INPUT_DIM](enc_input_buf.as_unsafe_any_origin())
    var enc_hpre_1 = tile[1, ENC_HIDDEN_DIM](enc_hpre_buf.as_unsafe_any_origin())
    var enc_hact_1 = tile[1, ENC_HIDDEN_DIM](enc_hact_buf.as_unsafe_any_origin())
    var enc_output_1 = tile[1, ENC_OUTPUT_DIM](enc_output_buf.as_unsafe_any_origin())

    var actions_buf = alloc[Scalar[dtype]](BATCH * SEQ_LEN)
    var obs_buf = alloc[Scalar[dtype]](BATCH * (SEQ_LEN + 1) * OBS_DIM)
    var prev_z_buf = alloc[Scalar[dtype]](BATCH * HIDDEN)

    # ── Train ────────────────────────────────────────────────────────────────
    print("\n  epoch | last_obs_loss | last_val_loss | wall_t (s)")
    print("  ------+---------------+---------------+------------")
    var rng = PhiloxRandom(seed=UInt64(7), offset=UInt64(0))
    var pc_step_num: Int = 0
    var enc_step_num: Int = 0
    var v_step_num: Int = 0
    var t0 = perf_counter_ns()

    for epoch in range(EPOCHS):
        var lr_scale = SCHED.lr_scale_at(epoch, EPOCHS)
        var last_loss: Float64 = 0.0
        var last_val_loss: Float64 = 0.0
        for batch_idx in range(N_BATCHES_PER_EPOCH):
            for b in range(BATCH):
                _gen_rollout_into[SEQ_LEN](
                    rng,
                    actions_buf.as_unsafe_any_origin(),
                    obs_buf.as_unsafe_any_origin(),
                    b * SEQ_LEN,
                    b * (SEQ_LEN + 1) * OBS_DIM,
                )
            # Termination tracking: for each rollout b, find first step t where
            # the unnormalized state would have terminated (|x|>2.4 or |θ|>0.2).
            # Rollouts in obs_buf are stored normalized; thresholds become |obs|>1.0
            # for components 0 (x) and 2 (theta). T_term = SEQ_LEN+1 if never terminates.
            for b in range(BATCH):
                var t_term: Int = SEQ_LEN + 1
                for t in range(SEQ_LEN + 1):
                    var x_n = Float64(
                        obs_buf[b * (SEQ_LEN + 1) * OBS_DIM + t * OBS_DIM + 0]
                    )
                    var th_n = Float64(
                        obs_buf[b * (SEQ_LEN + 1) * OBS_DIM + t * OBS_DIM + 2]
                    )
                    var x_a = x_n if x_n > 0.0 else -x_n
                    var th_a = th_n if th_n > 0.0 else -th_n
                    if x_a > 1.0 or th_a > 1.0:
                        t_term = t
                        break
                t_term_buf[b] = Scalar[dtype](Float64(t_term))

            unsafe_memset(prev_z_buf, 0, BATCH * HIDDEN)

            for t in range(0, SEQ_LEN + 1):
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

                ENC.forward[BATCH, dtype](
                    enc_params, enc_input, enc_hpre, enc_hact, enc_output
                )
                for b in range(BATCH):
                    for j in range(HIDDEN):
                        z_init_buf[b * HIDDEN + j] = enc_output_buf[
                            b * ENC_OUTPUT_DIM + j
                        ]

                BLOCK0.predict[BATCH, dtype](x_aug, params_b0, mu_z, a_aug)
                BLOCK0.eps_compute[BATCH, dtype](z_init, mu_z, eps_z)
                BLOCK1.predict[BATCH, dtype](z_init, params_b1, mu_obs, a_z)
                BLOCK1.eps_compute[BATCH, dtype](y_obs, mu_obs, eps_obs)

                var sum_sq: Float64 = 0.0
                for b in range(BATCH):
                    for d in range(OBS_DIM):
                        var e = Float64(eps_obs_buf[b * OBS_DIM + d])
                        sum_sq += e * e
                last_loss = 0.5 * sum_sq / Float64(BATCH)

                # ── Value head forward + targets + loss ──────────────────────
                # V_pred[b] = Σ_j W_val[j] · μ_z[b,j] + b_val
                # V_target[b] = MC return at step t, normalized by V_MAX.
                #   For step t in rollout b with termination at t_term:
                #     if t >= t_term: V_target = 0   (already terminated)
                #     else:            V_target = (1 − γ^(t_term − t)) / (1 − γ) / V_MAX
                # dV[b] = V_pred[b] − V_target[b]   (per-batch scalar)
                # ε_val[b, j] = −LAMBDA_VAL · dV[b] · W_val[j]   (signed for weight_grad)
                # value loss = (1/(2·BATCH)) Σ_b dV[b]²
                var v_loss_sum_sq: Float64 = 0.0
                for b in range(BATCH):
                    # V_pred
                    var v_pred: Float64 = Float64(v_params_buf[V_B_OFFSET])
                    for j in range(HIDDEN):
                        v_pred += Float64(
                            v_params_buf[V_W_OFFSET + j]
                        ) * Float64(mu_z_buf[b * HIDDEN + j])
                    # V_target
                    var t_term_b = Int(Float64(t_term_buf[b]))
                    var v_tgt: Float64 = 0.0
                    if t < t_term_b:
                        var n_left = t_term_b - t
                        v_tgt = (
                            (1.0 - GAMMA ** Float64(n_left))
                            / (1.0 - GAMMA)
                            / V_MAX
                        )
                    var dv = v_pred - v_tgt
                    v_loss_sum_sq += dv * dv
                    # ε_val[b, j] = −LAMBDA_VAL · dV[b] · W_val[j]
                    # weight_grad(ε_val, a_aug) then writes +LAMBDA_VAL · dL_val/dW_dyn
                    # into grads (sign matches PC rule output, ready for Adam).
                    for j in range(HIDDEN):
                        eps_val_buf[b * HIDDEN + j] = Scalar[dtype](
                            -LAMBDA_VAL
                            * dv
                            * Float64(v_params_buf[V_W_OFFSET + j])
                        )
                    # Value head grads accumulated below.
                last_val_loss = 0.5 * v_loss_sum_sq / Float64(BATCH)

                # Value head gradient — direct SGD, separate from dynamics grads.
                # dL/dW_val[j] = Σ_b dV[b] · μ_z[b, j]
                # dL/db_val    = Σ_b dV[b]
                unsafe_memset(v_grads_buf, 0, V_PARAM_SIZE)
                for b in range(BATCH):
                    var v_pred: Float64 = Float64(v_params_buf[V_B_OFFSET])
                    for j in range(HIDDEN):
                        v_pred += Float64(
                            v_params_buf[V_W_OFFSET + j]
                        ) * Float64(mu_z_buf[b * HIDDEN + j])
                    var t_term_b = Int(Float64(t_term_buf[b]))
                    var v_tgt: Float64 = 0.0
                    if t < t_term_b:
                        var n_left = t_term_b - t
                        v_tgt = (
                            (1.0 - GAMMA ** Float64(n_left))
                            / (1.0 - GAMMA)
                            / V_MAX
                        )
                    var dv = v_pred - v_tgt
                    for j in range(HIDDEN):
                        v_grads_buf[V_W_OFFSET + j] = Scalar[dtype](
                            Float64(v_grads_buf[V_W_OFFSET + j])
                            + dv * Float64(mu_z_buf[b * HIDDEN + j])
                        )
                    v_grads_buf[V_B_OFFSET] = Scalar[dtype](
                        Float64(v_grads_buf[V_B_OFFSET]) + dv
                    )

                unsafe_memset(pc_grads_buf, 0, NET.PARAM_SIZE)
                BLOCK0.weight_grad[BATCH, dtype](eps_z, a_aug, grads_b0)
                # Value contribution to block_0 dynamics grads.
                BLOCK0.weight_grad[BATCH, dtype](eps_val, a_aug, grads_b0_val)
                for k in range(BLOCK0.PARAM_SIZE):
                    pc_grads_buf[k] = Scalar[dtype](
                        Float64(pc_grads_buf[k]) + Float64(grads_b0_val_buf[k])
                    )
                BLOCK1.weight_grad[BATCH, dtype](eps_obs, a_z, grads_b1)

                BLOCK1.pull_back[BATCH, dtype](
                    eps_obs, params_b1, pull_back_out
                )
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

                # Value head update.
                clip_grad_norm[V_PARAM_SIZE, dtype](v_grads, GRAD_CLIP_NORM)
                v_step_num += 1
                OPT_VAL.step[V_PARAM_SIZE, dtype](
                    v_params,
                    v_grads,
                    v_opt_state,
                    v_opt_global,
                    v_step_num,
                    lr_scale=lr_scale,
                )

                for b in range(BATCH):
                    for j in range(HIDDEN):
                        prev_z_buf[b * HIDDEN + j] = z_init_buf[b * HIDDEN + j]

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            var elapsed = Float64(perf_counter_ns() - t0) / 1e9
            print(
                "    ep=",
                epoch,
                "  obs_loss=",
                last_loss,
                "  val_loss=",
                last_val_loss,
                "  lr_scale=",
                lr_scale,
                "  wall=",
                elapsed,
                "s",
            )

    var total_t = Float64(perf_counter_ns() - t0) / 1e9
    print("\n  total train time:", total_t, "s")

    # ────────────────────────────────────────────────────────────────────────
    # CEM eval.
    # ────────────────────────────────────────────────────────────────────────
    var cem_z_buf = alloc[Scalar[dtype]](N_SAMPLES * HIDDEN)
    var cem_z_next_buf = alloc[Scalar[dtype]](N_SAMPLES * HIDDEN)
    var cem_x_in_buf = alloc[Scalar[dtype]](N_SAMPLES * AUG_DIM)
    var cem_a_z_buf = alloc[Scalar[dtype]](N_SAMPLES * AUG_DIM)
    var cem_a_s_buf = alloc[Scalar[dtype]](N_SAMPLES * HIDDEN)
    var cem_obs_pred_buf = alloc[Scalar[dtype]](N_SAMPLES * OBS_DIM)
    var cem_actions_buf = alloc[Scalar[dtype]](N_SAMPLES * PLAN_HORIZON)
    var cem_x_in = tile[N_SAMPLES, AUG_DIM](cem_x_in_buf)
    var cem_a_z = tile[N_SAMPLES, AUG_DIM](cem_a_z_buf.as_unsafe_any_origin())
    var cem_z_next = tile[N_SAMPLES, HIDDEN](cem_z_next_buf.as_unsafe_any_origin())
    var cem_a_s = tile[N_SAMPLES, HIDDEN](cem_a_s_buf.as_unsafe_any_origin())
    var cem_obs_pred = tile[N_SAMPLES, OBS_DIM](cem_obs_pred_buf.as_unsafe_any_origin())

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

    var agent_z_buf = alloc[Scalar[dtype]](HIDDEN)

    print("\n  === CEM planning evaluation (no SGLD anywhere) ===")
    var eval_rng = PhiloxRandom(seed=UInt64(2027), offset=UInt64(0))
    var n_success: Int = 0
    var sum_steps_survived: Int = 0
    var t_eval_start = perf_counter_ns()

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
                        var x_norm = Float64(cem_obs_pred_buf[s * OBS_DIM + 0])
                        var th_norm = Float64(cem_obs_pred_buf[s * OBS_DIM + 2])
                        var a = Float64(cem_actions_buf[s * PLAN_HORIZON + h])
                        cem_scores[s] -= (
                            th_norm * th_norm
                            + POS_PENALTY * x_norm * x_norm
                            + ACTION_PENALTY * a * a
                        )
                    for s in range(N_SAMPLES):
                        for j in range(HIDDEN):
                            cem_z_buf[s * HIDDEN + j] = cem_z_next_buf[
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
                " steps (final |x|=",
                x if x > 0.0 else -x,
                " |θ|=",
                theta if theta > 0.0 else -theta,
                ")",
                " →",
                "PASS" if passed else "MISS",
            )
        else:
            print(
                "    ep=",
                ep,
                " : terminated at step ",
                terminated_at,
                " (|x|=",
                x if x > 0.0 else -x,
                " |θ|=",
                theta if theta > 0.0 else -theta,
                ")",
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

    print("\n  === Joint training (PCN dynamics + value head) summary ===")
    print(
        "  Solved",
        n_success,
        "/",
        N_EVAL_EPISODES,
        " (PCN world-only baseline: 5/5; MLP+BPTT world-only: 0/5)",
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
    v_params_buf.free()
    v_grads_buf.free()
    v_opt_state_buf.free()
    v_opt_global_buf.free()
    grads_b0_val_buf.free()
    eps_val_buf.free()
    t_term_buf.free()
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
