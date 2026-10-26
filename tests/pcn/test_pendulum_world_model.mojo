"""Pendulum world model — first real-env extension of the PCN world-model line.

Trains a stochastic action-conditioned tPC (Step 3 architecture) on Pendulum
trajectories with random actions. Evaluates:
  (1) **1-step teacher-forced prediction MSE**: at each step, filter z_t
      against the actual obs s_t, then predict s_{t+1} via feedforward
      block_0+block_1. Compares to persistence baseline (predict s = prev_s).
  (2) **Open-loop multi-step prediction MSE**: filter only the initial obs,
      then propagate latents forward without observation feedback. Tracks how
      prediction error grows across horizon.

Pendulum dynamics inlined here (with a Philox-seeded RNG) so the test is fully
reproducible without depending on Pendulum's global random_float64 RNG.

Observation space: 3D (cos θ, sin θ, ω). ω is normalized by max_speed=8 so all
input dims live in [-1, 1] (good for PCTanh). Action torque normalized by
max_torque=2.

Architecture:
    PCBlock[HIDDEN+1, HIDDEN, PCTanh]  # action-conditioned recurrence
    PCBlock[HIDDEN,   3,    PCTanh]    # 3-D emission

Run:
    pixi run mojo run -I . tests/pcn/test_pendulum_world_model.mojo
"""

from std.math import sqrt, log, cos, sin, pi
from std.memory import alloc, unsafe_memset
from std.random.philox import Random as PhiloxRandom
from std.time import perf_counter_ns
from noeira.core.tiles import Tile, tile

from noeira.nn.constants import DT as dtype
from noeira.experimental.pcn.pc_initializer import PCXavier
from noeira.experimental.pcn.pc_optimizer import PCAdam
from noeira.experimental.pcn import (
    PCBlock,
    PCSequential,
    PCTanh,
    PCTrainer,
)


# Pendulum physics
comptime PEND_G: Float64 = 10.0
comptime PEND_L: Float64 = 1.0
comptime PEND_M: Float64 = 1.0
comptime PEND_DT: Float64 = 0.05
comptime PEND_MAX_SPEED: Float64 = 8.0
comptime PEND_MAX_TORQUE: Float64 = 2.0

# World-model architecture
comptime BATCH = 32
comptime HIDDEN = 64
comptime ACTION_DIM = 1
comptime OBS_DIM = 3
comptime AUG_DIM = HIDDEN + ACTION_DIM
comptime SEQ_LEN = 10
comptime EVAL_HORIZON = 10
comptime EPOCHS = 80
comptime N_BATCHES_PER_EPOCH = 50
comptime T_INFER = 50
comptime LR_X: Float64 = 0.01
comptime ADAM_LR: Float64 = 0.001
comptime GRAD_CLIP_NORM: Float64 = 1.0  # global L2-norm clip; prevents Adam-driven W explosion in recurrent training

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
comptime OPT = PCAdam[LR=ADAM_LR]


def _angle_normalize(t: Float64) -> Float64:
    """Wrap angle into [-π, π]."""
    var x = (t + pi) - 2.0 * pi * Float64(Int((t + pi) / (2.0 * pi)))
    if x < 0.0:
        x += 2.0 * pi
    return x - pi


def _step_pendulum(
    mut theta: Float64, mut theta_dot: Float64, torque: Float64
) -> Tuple[Float64, Float64]:
    """One step of Pendulum dynamics. Returns (new_theta, new_theta_dot)."""
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


def _clip_grad_norm(
    grads: Tile[dtype, NET.PARAM_SIZE],
    max_norm: Float64,
) -> Float64:
    """Global L2-norm gradient clipping.

    If ‖g‖₂ > max_norm, rescales the entire gradient buffer so its L2 norm
    equals max_norm. Returns the (pre-clip) norm so callers can log it.
    Standard PyTorch-style `clip_grad_norm_`.
    """
    var sum_sq: Float64 = 0
    for i in range(NET.PARAM_SIZE):
        var g = Float64(grads.ptr[i])
        sum_sq += g * g
    var norm = sqrt(sum_sq)
    if norm > max_norm:
        var scale = Scalar[dtype](max_norm / norm)
        for i in range(NET.PARAM_SIZE):
            grads.ptr[i] = grads.ptr[i] * scale
    return norm


def _gen_rollout_into[
    SEQ_LEN_T: Int
](
    mut rng: PhiloxRandom,
    actions_buf: Pointer[Scalar[dtype], origin=MutAnyOrigin],
    obs_buf: Pointer[Scalar[dtype], origin=MutAnyOrigin],
    actions_offset: Int,
    obs_offset: Int,
):
    """Generate one Pendulum rollout into actions_buf[actions_offset..] and
    obs_buf[obs_offset..]. Initial state is random.

    actions[t] in [-1, 1] (normalized torque), t = 0..SEQ_LEN_T-1.
    obs[t, :] in [-1, 1]^3 (normalized [cos θ, sin θ, ω/max_speed]),
    t = 0..SEQ_LEN_T.
    """
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
    print("Pendulum world model — real-env extension")
    print("=" * 60)
    print(
        "  arch       : PCBlock[",
        AUG_DIM,
        ",",
        HIDDEN,
        ",PCTanh] → PCBlock[",
        HIDDEN,
        ",",
        OBS_DIM,
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
    print(
        "  T_INFER=",
        T_INFER,
        " LR_X=",
        LR_X,
        " ADAM_LR=",
        ADAM_LR,
        " GRAD_CLIP=",
        GRAD_CLIP_NORM,
    )
    print(
        "  env        : Pendulum (deterministic), obs=[cos θ, sin θ, ω/8],"
        " action ∈ [-1, 1]"
    )

    # ── Allocate net params + Adam state ──────────────────────────────────────
    var params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var grads_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var opt_state_buf = alloc[Scalar[dtype]](
        NET.PARAM_SIZE * OPT.STATE_PER_PARAM
    ).as_unsafe_any_origin()
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

    # Scratch buffers
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

    # Per-rollout actions/states scratch.
    var actions_buf = alloc[Scalar[dtype]](BATCH * SEQ_LEN).as_unsafe_any_origin()
    var obs_buf = alloc[Scalar[dtype]](BATCH * (SEQ_LEN + 1) * OBS_DIM).as_unsafe_any_origin()

    # ── Train ────────────────────────────────────────────────────────────────
    print("\n  epoch | last_step_loss | wall_t (s)")
    print("  ------+----------------+------------")
    var rng = PhiloxRandom(seed=UInt64(7), offset=UInt64(0))
    var step_num: Int = 0
    var t0 = perf_counter_ns()

    for epoch in range(EPOCHS):
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

            # Filter z_0 against s_0 (no action input — use zeros).
            unsafe_memset(x_in_buf, 0, BATCH * AUG_DIM)
            for b in range(BATCH):
                for d in range(OBS_DIM):
                    y_tgt_buf[b * OBS_DIM + d] = obs_buf[
                        b * (SEQ_LEN + 1) * OBS_DIM + 0 * OBS_DIM + d
                    ]
            _ = TRAINER.compute_grads_only[BATCH](
                params,
                grads,
                latents,
                mu_eps_buf,
                a_below_buf,
                z_below_buf,
                dx_buf,
                x_in,
                y_target,
                T_infer=T_INFER,
                lr_x=Scalar[dtype](LR_X),
            )
            _ = _clip_grad_norm(grads, GRAD_CLIP_NORM)
            step_num += 1
            OPT.step[NET.PARAM_SIZE, dtype](
                params, grads, opt_state, opt_global, step_num
            )
            for b in range(BATCH):
                for j in range(HIDDEN):
                    x_in_buf[b * AUG_DIM + j] = lat_buf[b * NET.LATENT_DIM + j]

            # Steps t = 1..SEQ_LEN: x_in = [prev_z, action_{t-1}], target = obs_t.
            for t in range(1, SEQ_LEN + 1):
                for b in range(BATCH):
                    x_in_buf[b * AUG_DIM + HIDDEN] = actions_buf[
                        b * SEQ_LEN + (t - 1)
                    ]
                    for d in range(OBS_DIM):
                        y_tgt_buf[b * OBS_DIM + d] = obs_buf[
                            b * (SEQ_LEN + 1) * OBS_DIM + t * OBS_DIM + d
                        ]
                var result = TRAINER.compute_grads_only[BATCH](
                    params,
                    grads,
                    latents,
                    mu_eps_buf,
                    a_below_buf,
                    z_below_buf,
                    dx_buf,
                    x_in,
                    y_target,
                    T_infer=T_INFER,
                    lr_x=Scalar[dtype](LR_X),
                )
                _ = _clip_grad_norm(grads, GRAD_CLIP_NORM)
                step_num += 1
                OPT.step[NET.PARAM_SIZE, dtype](
                    params, grads, opt_state, opt_global, step_num
                )
                last_loss = result.output_loss_final
                for b in range(BATCH):
                    for j in range(HIDDEN):
                        x_in_buf[b * AUG_DIM + j] = lat_buf[
                            b * NET.LATENT_DIM + j
                        ]

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            var elapsed = Float64(perf_counter_ns() - t0) / 1e9
            print(
                "    ",
                epoch,
                "  ",
                fit(String(last_loss), 11),
                "  ",
                fit(String(elapsed), 7),
            )

    var total_t = Float64(perf_counter_ns() - t0) / 1e9
    print("\n  total train time:", total_t, "s")

    # Per-block param views for eval feedforward
    comptime offset_b1 = NET._param_offset[1]()
    var params_b0 = tile[NET.block_types[0].PARAM_SIZE](params_buf)
    var params_b1 = tile[NET.block_types[1].PARAM_SIZE](params_buf + offset_b1)

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
    # At each step, filter z_t against actual s_t, predict s_{t+1} via
    # feedforward block_0+block_1. Compare to actual s_{t+1} and to persistence.
    print("\n  === Mode 1: 1-step teacher-forced prediction ===")
    print("  step | mse_total | mse_persist | per-dim model | per-dim persist")
    print("  -----+-----------+-------------+---------------+----------------")

    # Filter z_0 against s_0 (no action).
    unsafe_memset(x_in_buf, 0, BATCH * AUG_DIM)
    for b in range(N_EVAL_TRAJ):
        for d in range(OBS_DIM):
            y_tgt_buf[b * OBS_DIM + d] = obs_buf[
                b * (SEQ_LEN + 1) * OBS_DIM + 0 * OBS_DIM + d
            ]
    _ = TRAINER.compute_grads_only[BATCH](
        params,
        grads,
        latents,
        mu_eps_buf,
        a_below_buf,
        z_below_buf,
        dx_buf,
        x_in,
        y_target,
        T_infer=T_INFER,
        lr_x=Scalar[dtype](LR_X),
    )
    for b in range(N_EVAL_TRAJ):
        for j in range(HIDDEN):
            x_in_buf[b * AUG_DIM + j] = lat_buf[b * NET.LATENT_DIM + j]

    var mse_1step_total: Float64 = 0
    var mse_persist_total: Float64 = 0
    var mse_1step_per_dim_0: Float64 = 0
    var mse_1step_per_dim_1: Float64 = 0
    var mse_1step_per_dim_2: Float64 = 0
    var mse_persist_per_dim_0: Float64 = 0
    var mse_persist_per_dim_1: Float64 = 0
    var mse_persist_per_dim_2: Float64 = 0

    for t in range(1, SEQ_LEN + 1):
        # Build x_in[:, HIDDEN] = action_{t-1}.
        for b in range(N_EVAL_TRAJ):
            x_in_buf[b * AUG_DIM + HIDDEN] = actions_buf[b * SEQ_LEN + (t - 1)]

        # Predict (no settle): z_pred = block_0(x_in); s_pred = block_1(z_pred)
        NET.block_types[0].predict[BATCH, dtype](
            x_in, params_b0, z_pred, a_z_pred
        )
        NET.block_types[1].predict[BATCH, dtype](
            z_pred, params_b1, s_pred, a_s_pred
        )

        # MSE per dim
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
            var d_m_0 = Float64(s_pred_buf[b * OBS_DIM + 0]) - s_true_0
            var d_m_1 = Float64(s_pred_buf[b * OBS_DIM + 1]) - s_true_1
            var d_m_2 = Float64(s_pred_buf[b * OBS_DIM + 2]) - s_true_2
            step_mse_0 += d_m_0 * d_m_0
            step_mse_1 += d_m_1 * d_m_1
            step_mse_2 += d_m_2 * d_m_2
            var d_p_0 = s_prev_0 - s_true_0
            var d_p_1 = s_prev_1 - s_true_1
            var d_p_2 = s_prev_2 - s_true_2
            step_persist_0 += d_p_0 * d_p_0
            step_persist_1 += d_p_1 * d_p_1
            step_persist_2 += d_p_2 * d_p_2
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
        mse_1step_per_dim_0 += step_mse_0
        mse_1step_per_dim_1 += step_mse_1
        mse_1step_per_dim_2 += step_mse_2
        mse_persist_per_dim_0 += step_persist_0
        mse_persist_per_dim_1 += step_persist_1
        mse_persist_per_dim_2 += step_persist_2

        print(
            "    ",
            t,
            "  ",
            fit(String(mse_step), 9),
            "  ",
            fit(String(persist_step), 9),
            "  [",
            fit(String(step_mse_0), 6),
            ",",
            fit(String(step_mse_1), 6),
            ",",
            fit(String(step_mse_2), 6),
            "]",
            "  [",
            fit(String(step_persist_0), 6),
            ",",
            fit(String(step_persist_1), 6),
            ",",
            fit(String(step_persist_2), 6),
            "]",
        )

        # Filter z_t against actual s_t (teacher forcing).
        for b in range(N_EVAL_TRAJ):
            for d in range(OBS_DIM):
                y_tgt_buf[b * OBS_DIM + d] = obs_buf[
                    b * (SEQ_LEN + 1) * OBS_DIM + t * OBS_DIM + d
                ]
        _ = TRAINER.compute_grads_only[BATCH](
            params,
            grads,
            latents,
            mu_eps_buf,
            a_below_buf,
            z_below_buf,
            dx_buf,
            x_in,
            y_target,
            T_infer=T_INFER,
            lr_x=Scalar[dtype](LR_X),
        )
        for b in range(N_EVAL_TRAJ):
            for j in range(HIDDEN):
                x_in_buf[b * AUG_DIM + j] = lat_buf[b * NET.LATENT_DIM + j]

    var avg_mse_1step = mse_1step_total / Float64(SEQ_LEN)
    var avg_mse_persist = mse_persist_total / Float64(SEQ_LEN)
    print("\n  avg 1-step MSE :", avg_mse_1step)
    print("  avg persist MSE:", avg_mse_persist)
    print(
        "  ratio (model / persist):",
        avg_mse_1step / avg_mse_persist if avg_mse_persist > 0 else 1.0,
    )

    # ── Eval mode 2: open-loop multi-step prediction ─────────────────────────
    # Filter z_0 against s_0, then propagate forward via predict only.
    print("\n  === Mode 2: open-loop multi-step prediction ===")
    print("  step | mse_total (open-loop) | mse_persist | ratio")
    print("  -----+-----------------------+-------------+------")

    unsafe_memset(x_in_buf, 0, BATCH * AUG_DIM)
    for b in range(N_EVAL_TRAJ):
        for d in range(OBS_DIM):
            y_tgt_buf[b * OBS_DIM + d] = obs_buf[
                b * (SEQ_LEN + 1) * OBS_DIM + 0 * OBS_DIM + d
            ]
    _ = TRAINER.compute_grads_only[BATCH](
        params,
        grads,
        latents,
        mu_eps_buf,
        a_below_buf,
        z_below_buf,
        dx_buf,
        x_in,
        y_target,
        T_infer=T_INFER,
        lr_x=Scalar[dtype](LR_X),
    )
    for b in range(N_EVAL_TRAJ):
        for j in range(HIDDEN):
            x_in_buf[b * AUG_DIM + j] = lat_buf[b * NET.LATENT_DIM + j]

    var mse_openloop_total: Float64 = 0
    var mse_openloop_persist: Float64 = 0

    for t in range(1, EVAL_HORIZON + 1):
        for b in range(N_EVAL_TRAJ):
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

        # Open-loop: prev_hidden = z_pred (NOT filtered).
        for b in range(N_EVAL_TRAJ):
            for j in range(HIDDEN):
                x_in_buf[b * AUG_DIM + j] = z_pred_buf[b * HIDDEN + j]

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
    print(
        "  1-step MSE ratio  :",
        avg_mse_1step / avg_mse_persist if avg_mse_persist > 0 else 1.0,
        "  (want < 0.5)",
    )
    print(
        "  open-loop MSE ratio:",
        avg_mse_openloop / avg_mse_openloop_persist if avg_mse_openloop_persist
        > 0 else 1.0,
        "  (want < 1.0 — at least better than persistence)",
    )

    var pass_1step = (
        (avg_mse_1step / avg_mse_persist)
        < 0.5 if avg_mse_persist
        > 0 else False
    )
    var pass_openloop = (
        (avg_mse_openloop / avg_mse_openloop_persist)
        < 1.0 if avg_mse_openloop_persist
        > 0 else False
    )

    if pass_1step and pass_openloop:
        print(
            "\n  [PASS] Pendulum world model: 1-step MSE < 0.5×persistence,"
            " open-loop beats persistence"
        )
    else:
        if not pass_1step:
            print("\n  [FAIL] 1-step MSE ratio ≥ 0.5")
        if not pass_openloop:
            print(
                "\n  [FAIL] open-loop MSE ratio ≥ 1.0 — model worse than"
                " persistence"
            )
        raise Error("Pendulum world model test failed")

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
    obs_buf.free()
    z_pred_buf.free()
    a_z_pred_buf.free()
    s_pred_buf.free()
    a_s_pred_buf.free()
    print("=== Done ===")

from noeira.core.fmt import fit