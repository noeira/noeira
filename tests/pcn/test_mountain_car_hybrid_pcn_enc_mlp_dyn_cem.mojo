"""MountainCar Continuous CEM planning — Phase-1 ablation (Experiment 2).

Hybrid: PCN-trained encoder + MLP-trained transition + decoder.

Training:
- Phase 1 (PHASE1_EPOCHS): full PCN procedure (encoder + PC transition +
  PC decoder, joint amortized PC + SGLD). Produces a PCN-trained encoder.
  PC dynamics are discarded after this phase — only the encoder carries over.
- Phase 2 (PHASE2_EPOCHS): freeze encoder. Allocate fresh BPTT-style
  transition + decoder buffers (Xavier init). Train via K-step BPTT
  (K=SEQ_LEN, full-rollout) on the encoder-bootstrapped z_0. Encoder
  receives no gradient.

Eval: encoder forward only (no SGLD refinement) + BPTT-MLP imagination
in the CEM planner. Matches the MLP+BPTT baseline's eval procedure.

Hypothesis (Phase-1 Experiment 2, see docs/PCN_MBRL_PLAN.md):
- If the encoder is the source of PCN's win, hybrid should win like full PCN.
- If transition+decoder need PCN training too, hybrid degrades to MLP+BPTT.

Comparison only — pass criterion (≥ 3/5) reported but not enforced.

Run:
    pixi run mojo run -I . tests/pcn/test_mountain_car_hybrid_pcn_enc_mlp_dyn_cem.mojo
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
    PCTrainer,
    clip_grad_norm,
)


# MountainCar Continuous physics (Gymnasium defaults)
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
comptime K_BPTT = SEQ_LEN

# Phase splits — same total compute as the 100-epoch baselines.
comptime PHASE1_EPOCHS = 50
comptime PHASE2_EPOCHS = 50
comptime N_BATCHES_PER_EPOCH = 100

# PCN-specific (phase 1).
comptime T_REFINE_BOOTSTRAP = 30
comptime T_REFINE = 10
comptime LR_X: Float64 = 0.01
comptime ADAM_LR_PC: Float64 = 0.001
comptime ADAM_LR_ENC: Float64 = 0.001

# BPTT-specific (phase 2).
comptime ADAM_LR_BPTT: Float64 = 0.001

comptime WARMUP_EPOCHS = 5
comptime LR_MIN_SCALE: Float64 = 0.1
comptime GRAD_CLIP_NORM: Float64 = 1.0

comptime ENC_INPUT_DIM = HIDDEN + ACTION_DIM + OBS_DIM
comptime ENC_HIDDEN_DIM = 64
comptime ENC_OUTPUT_DIM = HIDDEN
comptime ENC = PCEncoder[ENC_INPUT_DIM, ENC_HIDDEN_DIM, ENC_OUTPUT_DIM]
comptime ENC_PARAM_SIZE = ENC.PARAM_SIZE

# BPTT layer sizes (single Linear+tanh per module).
comptime T_PARAM_SIZE = AUG_DIM * HIDDEN + HIDDEN
comptime D_PARAM_SIZE = HIDDEN * OBS_DIM + OBS_DIM
comptime T_W_OFFSET = 0
comptime T_B_OFFSET = AUG_DIM * HIDDEN
comptime D_W_OFFSET = 0
comptime D_B_OFFSET = HIDDEN * OBS_DIM

# CEM planning — identical to other tests.
comptime PLAN_HORIZON = 20
comptime N_SAMPLES = 128
comptime N_ELITES = 16
comptime N_CEM_ITERS = 2
comptime INITIAL_SIGMA: Float64 = 0.5
comptime MIN_SIGMA: Float64 = 0.05
comptime ACTION_PENALTY: Float64 = 0.001
comptime MAX_EPISODE_STEPS = 200
comptime N_EVAL_EPISODES = 5

# PCN compile-time wiring (phase 1).
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
comptime OPT_BPTT = PCAdam[LR=ADAM_LR_BPTT]
comptime SCHED_PHASE1 = CosineWarmupSchedule[
    WARMUP_EPOCHS=WARMUP_EPOCHS, MIN_SCALE=LR_MIN_SCALE
]
comptime SCHED_PHASE2 = CosineWarmupSchedule[
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


def _lt_backward_accum[
    BATCH_T: Int, IN: Int, OUT: Int
](
    params: Pointer[Scalar[dtype], origin=MutAnyOrigin],
    a: Pointer[Scalar[dtype], origin=MutAnyOrigin],
    d_mu: Pointer[Scalar[dtype], origin=MutAnyOrigin],
    d_W: Pointer[Scalar[dtype], origin=MutAnyOrigin],
    d_b: Pointer[Scalar[dtype], origin=MutAnyOrigin],
    d_x: Pointer[Scalar[dtype], origin=MutAnyOrigin],
):
    """Backward through `μ = tanh(x) @ W + b`. Accumulates d_W, d_b. Writes d_x.
    """
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
    print("MountainCar Continuous — Exp 2: hybrid PCN-encoder + MLP-dynamics")
    print("=" * 60)
    print(
        "  Phase 1    : full PCN for",
        PHASE1_EPOCHS,
        " epochs (encoder + PC dynamics)",
    )
    print(
        "  Phase 2    : frozen encoder + BPTT MLP dynamics for",
        PHASE2_EPOCHS,
        " epochs",
    )
    print("  K_BPTT     :", K_BPTT, " (full-rollout)")
    print("  Eval       : encoder forward only + BPTT-MLP imagination")

    # ── Phase-1 PC params + Adam state (PCSequential layout) ─────────────────
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

    # ── Encoder params + Adam state (carries from phase 1 to phase 2 frozen)
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

    # ── Phase-2 BPTT params (T = transition, D = decoder) ────────────────────
    var T_params_buf = alloc[Scalar[dtype]](T_PARAM_SIZE).as_unsafe_any_origin()
    var T_grads_buf = alloc[Scalar[dtype]](T_PARAM_SIZE).as_unsafe_any_origin()
    var T_opt_state_buf = alloc[Scalar[dtype]](
        T_PARAM_SIZE * OPT_BPTT.STATE_PER_PARAM
    ).as_unsafe_any_origin()
    var T_opt_global_buf = alloc[Scalar[dtype]](OPT_BPTT.GLOBAL_STATE_SIZE).as_unsafe_any_origin()
    var T_params = tile[T_PARAM_SIZE](T_params_buf)
    var T_grads = tile[T_PARAM_SIZE](T_grads_buf)
    var T_opt_state = tile[T_PARAM_SIZE, OPT_BPTT.STATE_PER_PARAM](T_opt_state_buf)
    var T_opt_global = tile[OPT_BPTT.GLOBAL_STATE_SIZE](T_opt_global_buf)

    var D_params_buf = alloc[Scalar[dtype]](D_PARAM_SIZE).as_unsafe_any_origin()
    var D_grads_buf = alloc[Scalar[dtype]](D_PARAM_SIZE).as_unsafe_any_origin()
    var D_opt_state_buf = alloc[Scalar[dtype]](
        D_PARAM_SIZE * OPT_BPTT.STATE_PER_PARAM
    ).as_unsafe_any_origin()
    var D_opt_global_buf = alloc[Scalar[dtype]](OPT_BPTT.GLOBAL_STATE_SIZE).as_unsafe_any_origin()
    var D_params = tile[D_PARAM_SIZE](D_params_buf)
    var D_grads = tile[D_PARAM_SIZE](D_grads_buf)
    var D_opt_state = tile[D_PARAM_SIZE, OPT_BPTT.STATE_PER_PARAM](D_opt_state_buf)
    var D_opt_global = tile[OPT_BPTT.GLOBAL_STATE_SIZE](D_opt_global_buf)

    # ── Shared scratch (PC-style for phase 1, sized to the union we need) ────
    var lat_buf = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var mu_eps_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_OUT_DIM).as_unsafe_any_origin()
    var a_below_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var z_below_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var dx_buf_raw = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    memset(lat_buf, 0, BATCH * NET.LATENT_DIM)
    var latents = tile[BATCH, NET.LATENT_DIM](lat_buf)
    var mu_eps_buf = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_eps_buf_raw)
    var a_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](a_below_buf_raw)
    var z_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](z_below_buf_raw)
    var dx_buf = tile[BATCH, NET.LATENT_DIM](dx_buf_raw)

    var x_in_buf = alloc[Scalar[dtype]](BATCH * AUG_DIM).as_unsafe_any_origin()
    var y_tgt_buf = alloc[Scalar[dtype]](BATCH * OBS_DIM).as_unsafe_any_origin()
    memset(x_in_buf, 0, BATCH * AUG_DIM)
    memset(y_tgt_buf, 0, BATCH * OBS_DIM)
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

    var enc_input_1 = tile[1, ENC_INPUT_DIM](enc_input_buf)
    var enc_hpre_1 = tile[1, ENC_HIDDEN_DIM](enc_hpre_buf)
    var enc_hact_1 = tile[1, ENC_HIDDEN_DIM](enc_hact_buf)
    var enc_output_1 = tile[1, ENC_OUTPUT_DIM](enc_output_buf)

    # ── BPTT cache (per-step activations, used in phase 2 only) ──────────────
    var cache_x_aug_buf = alloc[Scalar[dtype]](K_BPTT * BATCH * AUG_DIM).as_unsafe_any_origin()
    var cache_a_x_aug_buf = alloc[Scalar[dtype]](K_BPTT * BATCH * AUG_DIM).as_unsafe_any_origin()
    var cache_mu_z_next_buf = alloc[Scalar[dtype]](K_BPTT * BATCH * HIDDEN).as_unsafe_any_origin()
    var cache_a_z_next_buf = alloc[Scalar[dtype]](K_BPTT * BATCH * HIDDEN).as_unsafe_any_origin()
    var cache_mu_obs_buf = alloc[Scalar[dtype]](K_BPTT * BATCH * OBS_DIM).as_unsafe_any_origin()
    var d_mu_obs_buf = alloc[Scalar[dtype]](BATCH * OBS_DIM).as_unsafe_any_origin()
    var d_mu_z_next_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var d_z_from_next_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var d_x_aug_buf = alloc[Scalar[dtype]](BATCH * AUG_DIM).as_unsafe_any_origin()

    var actions_buf = alloc[Scalar[dtype]](BATCH * SEQ_LEN).as_unsafe_any_origin()
    var obs_buf = alloc[Scalar[dtype]](BATCH * (SEQ_LEN + 1) * OBS_DIM).as_unsafe_any_origin()

    # ────────────────────────────────────────────────────────────────────────
    # PHASE 1: full PCN training (encoder + PC dynamics, joint).
    # ────────────────────────────────────────────────────────────────────────
    print("\n  --- Phase 1 (PCN training) ---")
    print("  epoch | last_step_loss | wall_t (s)")
    print("  ------+----------------+------------")
    var rng = PhiloxRandom(seed=UInt64(7), offset=UInt64(0))
    var pc_step_num: Int = 0
    var enc_step_num: Int = 0
    var prev_z_buf = alloc[Scalar[dtype]](BATCH * HIDDEN).as_unsafe_any_origin()
    var t0 = perf_counter_ns()

    for epoch in range(PHASE1_EPOCHS):
        var lr_scale = SCHED_PHASE1.lr_scale_at(epoch, PHASE1_EPOCHS)
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
                        x_in_buf[b * AUG_DIM + j] = prev_z_buf[b * HIDDEN + j]
                    x_in_buf[b * AUG_DIM + HIDDEN] = act_val
                    for d in range(OBS_DIM):
                        y_tgt_buf[b * OBS_DIM + d] = obs_buf[
                            b * (SEQ_LEN + 1) * OBS_DIM + t * OBS_DIM + d
                        ]
                ENC.forward[BATCH, dtype](
                    enc_params, enc_input, enc_hpre, enc_hact, enc_output
                )
                for b in range(BATCH):
                    for j in range(HIDDEN):
                        lat_buf[b * NET.LATENT_DIM + j] = enc_output_buf[
                            b * ENC_OUTPUT_DIM + j
                        ]
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
                for b in range(BATCH):
                    for j in range(HIDDEN):
                        prev_z_buf[b * HIDDEN + j] = lat_buf[
                            b * NET.LATENT_DIM + j
                        ]

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == PHASE1_EPOCHS - 1:
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

    var phase1_time = Float64(perf_counter_ns() - t0) / 1e9
    print("  phase 1 wall:", phase1_time, "s")

    # ────────────────────────────────────────────────────────────────────────
    # PHASE 2: BPTT MLP dynamics, frozen encoder. Re-init T_params and D_params.
    # ────────────────────────────────────────────────────────────────────────
    print("\n  --- Phase 2 (BPTT MLP dynamics, frozen encoder) ---")
    print("  epoch | mean_loss | wall_t (s)")
    print("  ------+-----------+------------")

    # Re-init dynamics from scratch (Xavier). Adam state stays zero (already 0).
    memset(T_params_buf, 0, T_PARAM_SIZE)
    memset(T_grads_buf, 0, T_PARAM_SIZE)
    memset(T_opt_state_buf, 0, T_PARAM_SIZE * OPT_BPTT.STATE_PER_PARAM)
    memset(T_opt_global_buf, 0, OPT_BPTT.GLOBAL_STATE_SIZE)
    _xavier_init_layer(T_params_buf, AUG_DIM, HIDDEN, UInt64(7))
    memset(D_params_buf, 0, D_PARAM_SIZE)
    memset(D_grads_buf, 0, D_PARAM_SIZE)
    memset(D_opt_state_buf, 0, D_PARAM_SIZE * OPT_BPTT.STATE_PER_PARAM)
    memset(D_opt_global_buf, 0, OPT_BPTT.GLOBAL_STATE_SIZE)
    _xavier_init_layer(D_params_buf, HIDDEN, OBS_DIM, UInt64(8))

    var bptt_step_num: Int = 0
    var t1 = perf_counter_ns()

    for epoch in range(PHASE2_EPOCHS):
        var lr_scale = SCHED_PHASE2.lr_scale_at(epoch, PHASE2_EPOCHS)
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

            # t=0 encoder forward (frozen). z_0 = enc(0, 0, obs_0).
            for b in range(BATCH):
                for j in range(HIDDEN):
                    enc_input_buf[b * ENC_INPUT_DIM + j] = Scalar[dtype](0.0)
                enc_input_buf[b * ENC_INPUT_DIM + HIDDEN] = Scalar[dtype](0.0)
                for d in range(OBS_DIM):
                    enc_input_buf[
                        b * ENC_INPUT_DIM + HIDDEN + ACTION_DIM + d
                    ] = obs_buf[b * (SEQ_LEN + 1) * OBS_DIM + d]
            ENC.forward[BATCH, dtype](
                enc_params, enc_input, enc_hpre, enc_hact, enc_output
            )

            # Forward: roll z forward through K_BPTT transitions.
            for k in range(K_BPTT):
                for b in range(BATCH):
                    if k == 0:
                        for j in range(HIDDEN):
                            cache_x_aug_buf[
                                k * BATCH * AUG_DIM + b * AUG_DIM + j
                            ] = enc_output_buf[b * ENC_OUTPUT_DIM + j]
                    else:
                        for j in range(HIDDEN):
                            cache_x_aug_buf[
                                k * BATCH * AUG_DIM + b * AUG_DIM + j
                            ] = cache_mu_z_next_buf[
                                (k - 1) * BATCH * HIDDEN + b * HIDDEN + j
                            ]
                    cache_x_aug_buf[
                        k * BATCH * AUG_DIM + b * AUG_DIM + HIDDEN
                    ] = actions_buf[b * SEQ_LEN + k]
                _lt_forward[BATCH, AUG_DIM, HIDDEN](
                    T_params_buf,
                    cache_x_aug_buf + k * BATCH * AUG_DIM,
                    cache_a_x_aug_buf + k * BATCH * AUG_DIM,
                    cache_mu_z_next_buf + k * BATCH * HIDDEN,
                )
                _lt_forward[BATCH, HIDDEN, OBS_DIM](
                    D_params_buf,
                    cache_mu_z_next_buf + k * BATCH * HIDDEN,
                    cache_a_z_next_buf + k * BATCH * HIDDEN,
                    cache_mu_obs_buf + k * BATCH * OBS_DIM,
                )

            # Total loss.
            var sum_sq: Float64 = 0.0
            for k in range(K_BPTT):
                for b in range(BATCH):
                    for d in range(OBS_DIM):
                        var diff = Float64(
                            cache_mu_obs_buf[
                                k * BATCH * OBS_DIM + b * OBS_DIM + d
                            ]
                        ) - Float64(
                            obs_buf[
                                b * (SEQ_LEN + 1) * OBS_DIM
                                + (k + 1) * OBS_DIM
                                + d
                            ]
                        )
                        sum_sq += diff * diff
            last_loss = 0.5 * sum_sq / Float64(BATCH * K_BPTT)

            # Zero param grads, init d_z_from_next.
            memset(T_grads_buf, 0, T_PARAM_SIZE)
            memset(D_grads_buf, 0, D_PARAM_SIZE)
            memset(d_z_from_next_buf, 0, BATCH * HIDDEN)

            # Backward sweep: k = K-1 down to 0. NO encoder backward (frozen).
            for k_rev in range(K_BPTT):
                var k = K_BPTT - 1 - k_rev
                for b in range(BATCH):
                    for d in range(OBS_DIM):
                        d_mu_obs_buf[b * OBS_DIM + d] = Scalar[dtype](
                            Float64(
                                cache_mu_obs_buf[
                                    k * BATCH * OBS_DIM + b * OBS_DIM + d
                                ]
                            )
                            - Float64(
                                obs_buf[
                                    b * (SEQ_LEN + 1) * OBS_DIM
                                    + (k + 1) * OBS_DIM
                                    + d
                                ]
                            )
                        )
                _lt_backward_accum[BATCH, HIDDEN, OBS_DIM](
                    D_params_buf,
                    cache_a_z_next_buf + k * BATCH * HIDDEN,
                    d_mu_obs_buf,
                    D_grads_buf + D_W_OFFSET,
                    D_grads_buf + D_B_OFFSET,
                    d_mu_z_next_buf,
                )
                for b in range(BATCH):
                    for j in range(HIDDEN):
                        d_mu_z_next_buf[b * HIDDEN + j] = Scalar[dtype](
                            Float64(d_mu_z_next_buf[b * HIDDEN + j])
                            + Float64(d_z_from_next_buf[b * HIDDEN + j])
                        )
                _lt_backward_accum[BATCH, AUG_DIM, HIDDEN](
                    T_params_buf,
                    cache_a_x_aug_buf + k * BATCH * AUG_DIM,
                    d_mu_z_next_buf,
                    T_grads_buf + T_W_OFFSET,
                    T_grads_buf + T_B_OFFSET,
                    d_x_aug_buf,
                )
                for b in range(BATCH):
                    for j in range(HIDDEN):
                        d_z_from_next_buf[b * HIDDEN + j] = d_x_aug_buf[
                            b * AUG_DIM + j
                        ]

            # NO encoder Adam step (frozen).
            clip_grad_norm[T_PARAM_SIZE, dtype](T_grads, GRAD_CLIP_NORM)
            clip_grad_norm[D_PARAM_SIZE, dtype](D_grads, GRAD_CLIP_NORM)
            bptt_step_num += 1
            OPT_BPTT.step[T_PARAM_SIZE, dtype](
                T_params,
                T_grads,
                T_opt_state,
                T_opt_global,
                bptt_step_num,
                lr_scale=lr_scale,
            )
            OPT_BPTT.step[D_PARAM_SIZE, dtype](
                D_params,
                D_grads,
                D_opt_state,
                D_opt_global,
                bptt_step_num,
                lr_scale=lr_scale,
            )

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == PHASE2_EPOCHS - 1:
            var elapsed = Float64(perf_counter_ns() - t1) / 1e9
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

    var phase2_time = Float64(perf_counter_ns() - t1) / 1e9
    print("  phase 2 wall:", phase2_time, "s")
    print("  total train wall:", phase1_time + phase2_time, "s")

    # ────────────────────────────────────────────────────────────────────────
    # CEM imagination + eval (BPTT-style: encoder forward + MLP imagination).
    # ────────────────────────────────────────────────────────────────────────
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

    print("\n  === CEM planning evaluation (hybrid) ===")
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

        # Bootstrap: encoder forward only.
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
                        var pos_norm = Float64(cem_mu_obs_buf[s * OBS_DIM + 0])
                        var pos = pos_norm * MC_POS_HALF_RANGE + MC_POS_CENTER
                        if pos > cem_max_pos[s]:
                            cem_max_pos[s] = pos
                        var a = Float64(cem_actions_buf[s * PLAN_HORIZON + h])
                        cem_scores[s] -= ACTION_PENALTY * a * a
                    for s in range(N_SAMPLES):
                        for j in range(HIDDEN):
                            cem_z_buf[s * HIDDEN + j] = cem_mu_z_next_buf[
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

            # Filter: encoder only.
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

    print("\n  === Exp 2 (hybrid PCN-enc + MLP-dyn) summary ===")
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
    T_params_buf.free()
    T_grads_buf.free()
    T_opt_state_buf.free()
    T_opt_global_buf.free()
    D_params_buf.free()
    D_grads_buf.free()
    D_opt_state_buf.free()
    D_opt_global_buf.free()
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
    cache_x_aug_buf.free()
    cache_a_x_aug_buf.free()
    cache_mu_z_next_buf.free()
    cache_a_z_next_buf.free()
    cache_mu_obs_buf.free()
    d_mu_obs_buf.free()
    d_mu_z_next_buf.free()
    d_z_from_next_buf.free()
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
