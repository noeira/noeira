"""Pendulum world model — amortized PC (Tschantz 2023 hybrid PC).

Replaces the iterative SGLD inference with an encoder + few refinement steps:
    z_init = encoder([z_{t-1}, a_{t-1}, s_t])    # learned MLP, feedforward
    z_t    = z_init refined with K=5 SGLD steps  # local correction
    W's    updated via PC's local rule at refined z_t  (no backprop through W)
    encoder W's updated via MSE(encoder_out, refined_z_t.detach())  (standard backprop)

This addresses the two failure modes that vanilla iterative-PC hit on Pendulum:
  - **Lazy fit** is gone: encoder is forced to be a tight forward predictor.
  - **Recurrent inference instability** is gone: with K=5 instead of T_INFER=50,
    latents don't have time to diverge even if W's spectral radius drifts past 1.

PC's distinctives are preserved: local learning rule for the world model W's,
iterative refinement (just shorter), Bayesian framing.

Encoder is a 2-layer MLP, hand-rolled in this test file (test-local). If this
works, promote to a framework primitive.

Run:
    pixi run mojo run -I . tests/pcn/test_pendulum_amortized_pc.mojo
"""

from std.math import sqrt, log, cos, sin, tanh, pi
from std.memory import alloc, unsafe_memset
from std.random.philox import Random as PhiloxRandom
from std.time import perf_counter_ns
from noeira.core.tiles import tile

from noeira.nn.constants import DT as dtype
from noeira.experimental.pcn.pc_initializer import PCXavier
from noeira.experimental.pcn.pc_optimizer import PCAdam
from noeira.experimental.pcn.pc_optimizer import PCAdamW
from noeira.experimental.pcn.pc_scheduler import CosineWarmupSchedule
from noeira.experimental.pcn import (
    PCBlock,
    PCEncoder,
    PCSequential,
    PCTanh,
    PCTrainer,
    clip_grad_norm,
)


# Pendulum physics
comptime PEND_G: Float64 = 10.0
comptime PEND_L: Float64 = 1.0
comptime PEND_M: Float64 = 1.0
comptime PEND_DT: Float64 = 0.05
comptime PEND_MAX_SPEED: Float64 = 8.0
comptime PEND_MAX_TORQUE: Float64 = 2.0

# World-model architecture (PC)
comptime BATCH = 32
comptime HIDDEN = 64
comptime ACTION_DIM = 1
comptime OBS_DIM = 3
comptime AUG_DIM = HIDDEN + ACTION_DIM
comptime SEQ_LEN = 20
comptime EVAL_HORIZON = 10
comptime T_REFINE_BOOTSTRAP = 30  # Deeper refinement for the t=0 filter step
# (rare boundary case: prev_z=0, action=0)
comptime EPOCHS = 100
comptime N_BATCHES_PER_EPOCH = 100
comptime T_REFINE = 10
comptime WARMUP_EPOCHS = 5
comptime LR_MIN_SCALE: Float64 = 0.1  # K SGLD refinement steps after encoder init
comptime LR_X: Float64 = 0.01
comptime ADAM_LR_PC: Float64 = 0.001
comptime ADAM_LR_ENC: Float64 = 0.001
comptime GRAD_CLIP_NORM: Float64 = 1.0
comptime PC_WEIGHT_DECAY: Float64 = 0.01  # AdamW weight decay — regularizes decoder W to prevent unbounded ω predictions
# NOTE: spectral norm on the recurrence sub-block (σ_max(W_rec) ≤ 1) was tried
# and made Pendulum strictly worse (1-step ratio 0.35 → 3.0, open-loop 6.5 →
# 21.1). After tanh, the effective recurrence gain is σ_max · |tanh'| ≤ 1,
# which is strictly contractive — Pendulum's angular velocity needs gain > 1
# in some directions to track torque-driven dynamics. The technique is in the
# framework as `spectral_norm_clamp` for envs where it might help, but it
# does not fix the open-loop blowup here. See PCN_WORLD_MODEL_ROADMAP.md
# (Option B result, 2026-04-28) for the full diagnostic.

# Encoder architecture (framework PCEncoder, 2-layer MLP w/ tanh hidden)
comptime ENC_INPUT_DIM = HIDDEN + ACTION_DIM + OBS_DIM  # [prev_z, action, obs]
comptime ENC_HIDDEN_DIM = 64
comptime ENC_OUTPUT_DIM = HIDDEN
comptime ENC = PCEncoder[ENC_INPUT_DIM, ENC_HIDDEN_DIM, ENC_OUTPUT_DIM]
comptime ENC_PARAM_SIZE = ENC.PARAM_SIZE

# Eval
comptime N_EVAL_TRAJ = 32

comptime NET = PCSequential[
    PCBlock[AUG_DIM, HIDDEN, PCTanh],
    PCBlock[HIDDEN, OBS_DIM, PCTanh],
]
comptime TRAINER = PCTrainer[
    PCBlock[AUG_DIM, HIDDEN, PCTanh],
    PCBlock[HIDDEN, OBS_DIM, PCTanh],
    dtype=dtype,
]
comptime OPT_PC = PCAdam[LR=ADAM_LR_PC]
comptime OPT_ENC = PCAdam[LR=ADAM_LR_ENC]
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


def main() raises:
    print("=" * 60)
    print("Pendulum world model — amortized PC")
    print("=" * 60)
    print(
        "  PC arch    : PCBlock[",
        AUG_DIM,
        ",",
        HIDDEN,
        ",PCTanh] → PCBlock[",
        HIDDEN,
        ",",
        OBS_DIM,
        ",PCTanh]",
    )
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
    print("  Enc params :", ENC_PARAM_SIZE)
    print(
        "  BATCH=",
        BATCH,
        " SEQ_LEN=",
        SEQ_LEN,
        " EPOCHS=",
        EPOCHS,
        " T_REFINE=",
        T_REFINE,
    )
    print(
        "  PC_OPT=Adam(lr=", ADAM_LR_PC, ")  ENC_OPT=Adam(lr=", ADAM_LR_ENC, ")"
    )
    print(
        "  LR schedule: cosine warmup (W=",
        WARMUP_EPOCHS,
        ", min=",
        LR_MIN_SCALE,
        ")",
    )
    print("  LR_X=", LR_X, "  GRAD_CLIP=", GRAD_CLIP_NORM)

    # ── PC params + Adam state ────────────────────────────────────────────────
    var pc_params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var pc_grads_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var pc_opt_state_buf = alloc[Scalar[dtype]](
        NET.PARAM_SIZE * OPT_PC.STATE_PER_PARAM
    ).as_unsafe_any_origin()
    var pc_opt_global_buf = alloc[Scalar[dtype]](OPT_PC.GLOBAL_STATE_SIZE).as_unsafe_any_origin()
    unsafe_memset(pc_params_buf, 0, NET.PARAM_SIZE)
    unsafe_memset(pc_grads_buf, 0, NET.PARAM_SIZE)
    unsafe_memset(pc_opt_state_buf, 0, NET.PARAM_SIZE * OPT_PC.STATE_PER_PARAM)
    unsafe_memset(pc_opt_global_buf, 0, OPT_PC.GLOBAL_STATE_SIZE)
    var pc_params = tile[NET.PARAM_SIZE](pc_params_buf)
    var pc_grads = tile[NET.PARAM_SIZE](pc_grads_buf)
    var pc_opt_state = tile[NET.PARAM_SIZE, OPT_PC.STATE_PER_PARAM](pc_opt_state_buf)
    var pc_opt_global = tile[OPT_PC.GLOBAL_STATE_SIZE](pc_opt_global_buf)
    NET.pc_init_params[PCXavier, dtype](pc_params)

    # Per-block PC param views — used during eval (open-loop rollout).
    comptime offset_b1 = NET._param_offset[1]()
    var params_b0 = tile[NET.block_types[0].PARAM_SIZE](pc_params_buf)
    var params_b1 = tile[NET.block_types[1].PARAM_SIZE](pc_params_buf + offset_b1)

    # ── Encoder params + Adam state ───────────────────────────────────────────
    var enc_params_buf = alloc[Scalar[dtype]](ENC_PARAM_SIZE).as_unsafe_any_origin()
    var enc_grads_buf = alloc[Scalar[dtype]](ENC_PARAM_SIZE).as_unsafe_any_origin()
    var enc_opt_state_buf = alloc[Scalar[dtype]](
        ENC_PARAM_SIZE * OPT_ENC.STATE_PER_PARAM
    ).as_unsafe_any_origin()
    var enc_opt_global_buf = alloc[Scalar[dtype]](OPT_ENC.GLOBAL_STATE_SIZE).as_unsafe_any_origin()
    unsafe_memset(enc_params_buf, 0, ENC_PARAM_SIZE)
    unsafe_memset(enc_grads_buf, 0, ENC_PARAM_SIZE)
    unsafe_memset(enc_opt_state_buf, 0, ENC_PARAM_SIZE * OPT_ENC.STATE_PER_PARAM)
    unsafe_memset(enc_opt_global_buf, 0, OPT_ENC.GLOBAL_STATE_SIZE)

    var enc_params = tile[ENC_PARAM_SIZE](enc_params_buf)
    var enc_grads = tile[ENC_PARAM_SIZE](enc_grads_buf)
    var enc_opt_state = tile[ENC_PARAM_SIZE, OPT_ENC.STATE_PER_PARAM](enc_opt_state_buf)
    var enc_opt_global = tile[OPT_ENC.GLOBAL_STATE_SIZE](enc_opt_global_buf)
    ENC.xavier_init[dtype](enc_params, UInt64(123))

    # ── PC scratch ────────────────────────────────────────────────────────────
    var lat_buf = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var mu_eps_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_OUT_DIM).as_unsafe_any_origin()
    var a_below_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var z_below_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var dx_buf_raw = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    unsafe_memset(lat_buf, 0, BATCH * NET.LATENT_DIM)

    var latents = tile[BATCH, NET.LATENT_DIM](lat_buf)
    var mu_eps_buf = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_eps_buf_raw)
    var a_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](a_below_buf_raw)
    var z_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](z_below_buf_raw)
    var dx_buf = tile[BATCH, NET.LATENT_DIM](dx_buf_raw)

    var x_in_buf = alloc[Scalar[dtype]](BATCH * AUG_DIM).as_unsafe_any_origin()
    var y_tgt_buf = alloc[Scalar[dtype]](BATCH * OBS_DIM).as_unsafe_any_origin()
    unsafe_memset(x_in_buf, 0, BATCH * AUG_DIM)
    unsafe_memset(y_tgt_buf, 0, BATCH * OBS_DIM)
    var x_in = tile[BATCH, AUG_DIM](x_in_buf)
    var y_target = tile[BATCH, OBS_DIM](y_tgt_buf)

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

    # ── Per-rollout actions/states scratch ───────────────────────────────────
    var actions_buf = alloc[Scalar[dtype]](BATCH * SEQ_LEN).as_unsafe_any_origin()
    var obs_buf = alloc[Scalar[dtype]](BATCH * (SEQ_LEN + 1) * OBS_DIM).as_unsafe_any_origin()

    # ── Train ────────────────────────────────────────────────────────────────
    print("\n  epoch | last_step_loss | wall_t (s)")
    print("  ------+----------------+------------")
    var rng = PhiloxRandom(seed=UInt64(7), offset=UInt64(0))
    var pc_step_num: Int = 0
    var enc_step_num: Int = 0
    var prev_z_buf = alloc[Scalar[dtype]](
        BATCH * HIDDEN
    ).as_unsafe_any_origin()  # carries z_{t-1} between time steps
    var t0 = perf_counter_ns()

    for epoch in range(EPOCHS):
        # Cosine warmup schedule: linear 0→1 over WARMUP_EPOCHS, then cosine
        # decay to LR_MIN_SCALE. Flattens late-epoch updates so Adam can't
        # destabilize the recurrent training.
        var lr_scale = SCHED.lr_scale_at(epoch, EPOCHS)
        var last_loss: Float64 = 0.0
        for batch_idx in range(N_BATCHES_PER_EPOCH):
            # Generate BATCH rollouts on host.
            for b in range(BATCH):
                _gen_rollout_into[SEQ_LEN](
                    rng,
                    actions_buf,
                    obs_buf,
                    b * SEQ_LEN,
                    b * (SEQ_LEN + 1) * OBS_DIM,
                )

            # prev_z = 0 at the start of each rollout.
            unsafe_memset(prev_z_buf, 0, BATCH * HIDDEN)

            # Per time step t = 0..SEQ_LEN: encode → refine → update W's.
            # At t=0 there's no prior action; use action=0 in the encoder input.
            for t in range(0, SEQ_LEN + 1):
                # Build encoder input: [prev_z, action_{t-1} or 0, obs_t]
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
                    # PC trainer's x_in = [prev_z, action]
                    for j in range(HIDDEN):
                        x_in_buf[b * AUG_DIM + j] = prev_z_buf[b * HIDDEN + j]
                    x_in_buf[b * AUG_DIM + HIDDEN] = act_val
                    # PC trainer's y_target = obs_t
                    for d in range(OBS_DIM):
                        y_tgt_buf[b * OBS_DIM + d] = obs_buf[
                            b * (SEQ_LEN + 1) * OBS_DIM + t * OBS_DIM + d
                        ]

                # Encoder forward → write encoder_out into latents.
                ENC.forward[BATCH, dtype](
                    enc_params, enc_input, enc_hpre, enc_hact, enc_output
                )
                for b in range(BATCH):
                    for j in range(HIDDEN):
                        lat_buf[b * NET.LATENT_DIM + j] = enc_output_buf[
                            b * ENC_OUTPUT_DIM + j
                        ]

                # T_REFINE SGLD steps (or T_REFINE_BOOTSTRAP at t=0) + PC W
                # gradients. The t=0 boundary case (prev_z=0, action=0) is
                # rare in training (1/(SEQ_LEN+1) of transitions) and the
                # encoder's initial guess can be poor; deeper refinement at
                # t=0 lets settling correct it before the W gradient is
                # taken at the (less-noisy) energy minimum.
                var T_refine_step = T_REFINE_BOOTSTRAP if t == 0 else T_REFINE
                var result = TRAINER.compute_grads_from_latents[BATCH](
                    pc_params,
                    pc_grads,
                    latents,
                    mu_eps_buf,
                    a_below_buf,
                    z_below_buf,
                    dx_buf,
                    x_in,
                    y_target,
                    T_infer=T_refine_step,
                    lr_x=Scalar[dtype](LR_X),
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
                last_loss = result.output_loss_final

                # Encoder gradient: dz = encoder_out - settled_z (stop-gradient on settled).
                for b in range(BATCH):
                    for j in range(HIDDEN):
                        enc_dz_buf[b * ENC_OUTPUT_DIM + j] = (
                            enc_output_buf[b * ENC_OUTPUT_DIM + j]
                            - lat_buf[b * NET.LATENT_DIM + j]
                        )
                ENC.backward[BATCH, dtype](
                    enc_params, enc_input, enc_hact, enc_dz, enc_grads
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

                # prev_z = settled z_t (post-refinement).
                for b in range(BATCH):
                    for j in range(HIDDEN):
                        prev_z_buf[b * HIDDEN + j] = lat_buf[
                            b * NET.LATENT_DIM + j
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

    # Eval feedforward scratch
    var z_pred_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var a_z_pred_buf = alloc[Scalar[dtype]](BATCH * AUG_DIM).as_unsafe_any_origin()
    var s_pred_buf = alloc[Scalar[dtype]](BATCH * OBS_DIM).as_unsafe_any_origin()
    var a_s_pred_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var z_pred = tile[BATCH, HIDDEN](z_pred_buf)
    var a_z_pred = tile[BATCH, AUG_DIM](a_z_pred_buf)
    var s_pred = tile[BATCH, OBS_DIM](s_pred_buf)
    var a_s_pred = tile[BATCH, HIDDEN](a_s_pred_buf)

    # ── Generate held-out trajectories ───────────────────────────────────────
    var eval_rng = PhiloxRandom(seed=UInt64(2026), offset=UInt64(0))
    for b in range(N_EVAL_TRAJ):
        _gen_rollout_into[SEQ_LEN](
            eval_rng,
            actions_buf,
            obs_buf,
            b * SEQ_LEN,
            b * (SEQ_LEN + 1) * OBS_DIM,
        )

    # ── Eval mode 1: 1-step teacher-forced prediction ────────────────────────
    # At each step, encode/filter z_t against actual s_t, predict s_{t+1} via
    # feedforward block_0+block_1 (PC recurrence). Compare to actual s_{t+1}.
    print("\n  === Mode 1: 1-step teacher-forced prediction ===")
    print(
        "  Per-dim cos/sin/ω: model_total | persist_total | model[per-dim] |"
        " persist[per-dim]"
    )

    unsafe_memset(prev_z_buf, 0, BATCH * HIDDEN)

    # Initial filter: encode + refine at t=0 (bootstrap).
    for b in range(N_EVAL_TRAJ):
        for j in range(HIDDEN):
            enc_input_buf[b * ENC_INPUT_DIM + j] = prev_z_buf[b * HIDDEN + j]
        enc_input_buf[b * ENC_INPUT_DIM + HIDDEN] = Scalar[dtype](0.0)
        for d in range(OBS_DIM):
            enc_input_buf[
                b * ENC_INPUT_DIM + HIDDEN + ACTION_DIM + d
            ] = obs_buf[b * (SEQ_LEN + 1) * OBS_DIM + 0 + d]
        for j in range(HIDDEN):
            x_in_buf[b * AUG_DIM + j] = prev_z_buf[b * HIDDEN + j]
        x_in_buf[b * AUG_DIM + HIDDEN] = Scalar[dtype](0.0)
        for d in range(OBS_DIM):
            y_tgt_buf[b * OBS_DIM + d] = obs_buf[
                b * (SEQ_LEN + 1) * OBS_DIM + 0 + d
            ]
    ENC.forward[BATCH, dtype](
        enc_params, enc_input, enc_hpre, enc_hact, enc_output
    )
    for b in range(N_EVAL_TRAJ):
        for j in range(HIDDEN):
            lat_buf[b * NET.LATENT_DIM + j] = enc_output_buf[
                b * ENC_OUTPUT_DIM + j
            ]
    # Bootstrap filter (t=0): deeper refinement than per-step.
    _ = TRAINER.compute_grads_from_latents[BATCH](
        pc_params,
        pc_grads,
        latents,
        mu_eps_buf,
        a_below_buf,
        z_below_buf,
        dx_buf,
        x_in,
        y_target,
        T_infer=T_REFINE_BOOTSTRAP,
        lr_x=Scalar[dtype](LR_X),
    )
    for b in range(N_EVAL_TRAJ):
        for j in range(HIDDEN):
            prev_z_buf[b * HIDDEN + j] = lat_buf[b * NET.LATENT_DIM + j]

    var mse_1step_total: Float64 = 0
    var mse_persist_total: Float64 = 0

    for t in range(1, SEQ_LEN + 1):
        # Build x_in = [prev_z, action_{t-1}]
        for b in range(N_EVAL_TRAJ):
            for j in range(HIDDEN):
                x_in_buf[b * AUG_DIM + j] = prev_z_buf[b * HIDDEN + j]
            x_in_buf[b * AUG_DIM + HIDDEN] = actions_buf[b * SEQ_LEN + (t - 1)]

        # 1-step prediction via PC feedforward (no settle): z_pred = block_0(x_in); s_pred = block_1(z_pred)
        NET.block_types[0].predict[BATCH, dtype](
            x_in, params_b0, z_pred, a_z_pred
        )
        NET.block_types[1].predict[BATCH, dtype](
            z_pred, params_b1, s_pred, a_s_pred
        )

        # MSE
        var step_mse_0: Float64 = 0
        var step_mse_1: Float64 = 0
        var step_mse_2: Float64 = 0
        var step_persist_0: Float64 = 0
        var step_persist_1: Float64 = 0
        var step_persist_2: Float64 = 0
        for b in range(N_EVAL_TRAJ):
            var s_true_0 = Float64(
                obs_buf[b * (SEQ_LEN + 1) * OBS_DIM + t * OBS_DIM + 0]
            )
            var s_true_1 = Float64(
                obs_buf[b * (SEQ_LEN + 1) * OBS_DIM + t * OBS_DIM + 1]
            )
            var s_true_2 = Float64(
                obs_buf[b * (SEQ_LEN + 1) * OBS_DIM + t * OBS_DIM + 2]
            )
            var s_prev_0 = Float64(
                obs_buf[b * (SEQ_LEN + 1) * OBS_DIM + (t - 1) * OBS_DIM + 0]
            )
            var s_prev_1 = Float64(
                obs_buf[b * (SEQ_LEN + 1) * OBS_DIM + (t - 1) * OBS_DIM + 1]
            )
            var s_prev_2 = Float64(
                obs_buf[b * (SEQ_LEN + 1) * OBS_DIM + (t - 1) * OBS_DIM + 2]
            )
            var d0 = Float64(s_pred_buf[b * OBS_DIM + 0]) - s_true_0
            var d1 = Float64(s_pred_buf[b * OBS_DIM + 1]) - s_true_1
            var d2 = Float64(s_pred_buf[b * OBS_DIM + 2]) - s_true_2
            step_mse_0 += d0 * d0
            step_mse_1 += d1 * d1
            step_mse_2 += d2 * d2
            var p0 = s_prev_0 - s_true_0
            var p1 = s_prev_1 - s_true_1
            var p2 = s_prev_2 - s_true_2
            step_persist_0 += p0 * p0
            step_persist_1 += p1 * p1
            step_persist_2 += p2 * p2
        step_mse_0 /= Float64(N_EVAL_TRAJ)
        step_mse_1 /= Float64(N_EVAL_TRAJ)
        step_mse_2 /= Float64(N_EVAL_TRAJ)
        step_persist_0 /= Float64(N_EVAL_TRAJ)
        step_persist_1 /= Float64(N_EVAL_TRAJ)
        step_persist_2 /= Float64(N_EVAL_TRAJ)
        var mse_step = step_mse_0 + step_mse_1 + step_mse_2
        var persist_step = step_persist_0 + step_persist_1 + step_persist_2
        mse_1step_total += mse_step
        mse_persist_total += persist_step
        # Untruncated print — Mojo renders very small Float64 in scientific
        # notation (e.g. "1.12e-04"); truncating to fixed bytes can hide the
        # exponent suffix and badly mislead diagnosis.
        print(
            "    t=",
            t,
            " | total=",
            mse_step,
            " | persist=",
            persist_step,
            " | per-dim=[",
            step_mse_0,
            ", ",
            step_mse_1,
            ", ",
            step_mse_2,
            "]",
            " | persist=[",
            step_persist_0,
            ", ",
            step_persist_1,
            ", ",
            step_persist_2,
            "]",
        )

        # Filter z_t: encode against actual s_t, refine, save.
        for b in range(N_EVAL_TRAJ):
            for j in range(HIDDEN):
                enc_input_buf[b * ENC_INPUT_DIM + j] = prev_z_buf[
                    b * HIDDEN + j
                ]
            enc_input_buf[b * ENC_INPUT_DIM + HIDDEN] = actions_buf[
                b * SEQ_LEN + (t - 1)
            ]
            for d in range(OBS_DIM):
                enc_input_buf[
                    b * ENC_INPUT_DIM + HIDDEN + ACTION_DIM + d
                ] = obs_buf[b * (SEQ_LEN + 1) * OBS_DIM + t * OBS_DIM + d]
            for d in range(OBS_DIM):
                y_tgt_buf[b * OBS_DIM + d] = obs_buf[
                    b * (SEQ_LEN + 1) * OBS_DIM + t * OBS_DIM + d
                ]
        ENC.forward[BATCH, dtype](
            enc_params, enc_input, enc_hpre, enc_hact, enc_output
        )
        for b in range(N_EVAL_TRAJ):
            for j in range(HIDDEN):
                lat_buf[b * NET.LATENT_DIM + j] = enc_output_buf[
                    b * ENC_OUTPUT_DIM + j
                ]
        _ = TRAINER.compute_grads_from_latents[BATCH](
            pc_params,
            pc_grads,
            latents,
            mu_eps_buf,
            a_below_buf,
            z_below_buf,
            dx_buf,
            x_in,
            y_target,
            T_infer=T_REFINE,
            lr_x=Scalar[dtype](LR_X),
        )
        for b in range(N_EVAL_TRAJ):
            for j in range(HIDDEN):
                prev_z_buf[b * HIDDEN + j] = lat_buf[b * NET.LATENT_DIM + j]

    var avg_mse_1step = mse_1step_total / Float64(SEQ_LEN)
    var avg_mse_persist = mse_persist_total / Float64(SEQ_LEN)
    print("\n  avg 1-step MSE :", avg_mse_1step)
    print("  avg persist MSE:", avg_mse_persist)
    print(
        "  ratio (model / persist):",
        avg_mse_1step / avg_mse_persist if avg_mse_persist > 0 else 1.0,
    )

    # ── Eval mode 2: open-loop multi-step prediction ─────────────────────────
    print("\n  === Mode 2: open-loop multi-step prediction ===")
    print("  step | mse_total (open-loop) | mse_persist | ratio")
    print("  -----+-----------------------+-------------+------")

    unsafe_memset(prev_z_buf, 0, BATCH * HIDDEN)

    # Initial filter at t=0
    for b in range(N_EVAL_TRAJ):
        for j in range(HIDDEN):
            enc_input_buf[b * ENC_INPUT_DIM + j] = prev_z_buf[b * HIDDEN + j]
        enc_input_buf[b * ENC_INPUT_DIM + HIDDEN] = Scalar[dtype](0.0)
        for d in range(OBS_DIM):
            enc_input_buf[
                b * ENC_INPUT_DIM + HIDDEN + ACTION_DIM + d
            ] = obs_buf[b * (SEQ_LEN + 1) * OBS_DIM + 0 + d]
        for j in range(HIDDEN):
            x_in_buf[b * AUG_DIM + j] = prev_z_buf[b * HIDDEN + j]
        x_in_buf[b * AUG_DIM + HIDDEN] = Scalar[dtype](0.0)
        for d in range(OBS_DIM):
            y_tgt_buf[b * OBS_DIM + d] = obs_buf[
                b * (SEQ_LEN + 1) * OBS_DIM + 0 + d
            ]
    ENC.forward[BATCH, dtype](
        enc_params, enc_input, enc_hpre, enc_hact, enc_output
    )
    for b in range(N_EVAL_TRAJ):
        for j in range(HIDDEN):
            lat_buf[b * NET.LATENT_DIM + j] = enc_output_buf[
                b * ENC_OUTPUT_DIM + j
            ]
    # Bootstrap filter (t=0): deeper refinement than per-step.
    _ = TRAINER.compute_grads_from_latents[BATCH](
        pc_params,
        pc_grads,
        latents,
        mu_eps_buf,
        a_below_buf,
        z_below_buf,
        dx_buf,
        x_in,
        y_target,
        T_infer=T_REFINE_BOOTSTRAP,
        lr_x=Scalar[dtype](LR_X),
    )
    for b in range(N_EVAL_TRAJ):
        for j in range(HIDDEN):
            prev_z_buf[b * HIDDEN + j] = lat_buf[b * NET.LATENT_DIM + j]

    var mse_openloop_total: Float64 = 0
    var mse_openloop_persist: Float64 = 0

    for t in range(1, EVAL_HORIZON + 1):
        for b in range(N_EVAL_TRAJ):
            for j in range(HIDDEN):
                x_in_buf[b * AUG_DIM + j] = prev_z_buf[b * HIDDEN + j]
            x_in_buf[b * AUG_DIM + HIDDEN] = actions_buf[b * SEQ_LEN + (t - 1)]
        NET.block_types[0].predict[BATCH, dtype](
            x_in, params_b0, z_pred, a_z_pred
        )
        NET.block_types[1].predict[BATCH, dtype](
            z_pred, params_b1, s_pred, a_s_pred
        )

        var step_mse: Float64 = 0
        var step_persist: Float64 = 0
        for b in range(N_EVAL_TRAJ):
            for d in range(OBS_DIM):
                var s_true = Float64(
                    obs_buf[b * (SEQ_LEN + 1) * OBS_DIM + t * OBS_DIM + d]
                )
                var s_prev = Float64(
                    obs_buf[b * (SEQ_LEN + 1) * OBS_DIM + (t - 1) * OBS_DIM + d]
                )
                var d_m = Float64(s_pred_buf[b * OBS_DIM + d]) - s_true
                step_mse += d_m * d_m
                var d_p = s_prev - s_true
                step_persist += d_p * d_p
        step_mse /= Float64(N_EVAL_TRAJ)
        step_persist /= Float64(N_EVAL_TRAJ)
        mse_openloop_total += step_mse
        mse_openloop_persist += step_persist
        print(
            "    ",
            t,
            "  ",
            fit(String(step_mse), 9),
            "  ",
            fit(String(step_persist), 9),
            "  ",
            fit(String(step_mse / step_persist if step_persist > 0 else 1.0), 6),
        )

        # Open-loop: prev_z = z_pred (predicted, NOT filtered).
        for b in range(N_EVAL_TRAJ):
            for j in range(HIDDEN):
                prev_z_buf[b * HIDDEN + j] = z_pred_buf[b * HIDDEN + j]

    var avg_mse_openloop = mse_openloop_total / Float64(EVAL_HORIZON)
    var avg_mse_openloop_persist = mse_openloop_persist / Float64(EVAL_HORIZON)
    print("\n  avg open-loop MSE :", avg_mse_openloop)
    print("  avg open-loop persist:", avg_mse_openloop_persist)
    print(
        "  ratio (model / persist):",
        avg_mse_openloop / avg_mse_openloop_persist if avg_mse_openloop_persist
        > 0 else 1.0,
    )

    # ── Pass criteria ─────────────────────────────────────────────────────────
    print("\n  === Summary ===")
    var ratio_1step = (
        avg_mse_1step / avg_mse_persist if avg_mse_persist > 0 else 1.0
    )
    var ratio_openloop = (
        avg_mse_openloop / avg_mse_openloop_persist if avg_mse_openloop_persist
        > 0 else 1.0
    )
    print("  1-step MSE ratio  :", ratio_1step, "  (want < 0.5)")
    print("  open-loop MSE ratio:", ratio_openloop, "  (want < 1.0)")

    var pass_1step = ratio_1step < 0.5
    var pass_openloop = ratio_openloop < 1.0

    if pass_1step and pass_openloop:
        print(
            "\n  [PASS] amortized PC: 1-step MSE < 0.5×persistence, open-loop"
            " beats persistence"
        )
    else:
        if not pass_1step:
            print("\n  [FAIL] 1-step MSE ratio ≥ 0.5")
        if not pass_openloop:
            print(
                "\n  [FAIL] open-loop MSE ratio ≥ 1.0 — model worse than"
                " persistence"
            )
        raise Error("amortized PC test failed")

    # cleanup
    pc_params_buf.free()
    pc_grads_buf.free()
    pc_opt_state_buf.free()
    pc_opt_global_buf.free()
    enc_params_buf.free()
    enc_grads_buf.free()
    enc_opt_state_buf.free()
    enc_opt_global_buf.free()
    lat_buf.free()
    mu_eps_buf_raw.free()
    a_below_buf_raw.free()
    z_below_buf_raw.free()
    dx_buf_raw.free()
    x_in_buf.free()
    y_tgt_buf.free()
    enc_input_buf.free()
    enc_hpre_buf.free()
    enc_hact_buf.free()
    enc_output_buf.free()
    enc_dz_buf.free()
    actions_buf.free()
    obs_buf.free()
    prev_z_buf.free()
    z_pred_buf.free()
    a_z_pred_buf.free()
    s_pred_buf.free()
    a_s_pred_buf.free()
    print("=== Done ===")

from noeira.core.fmt import fit