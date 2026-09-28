"""MBPO training on HalfCheetah (GPU) via the new `MBPOAgent` facade.

GPU sibling of `mbpo_half_cheetah_training.mojo` (CPU). Same deep_agents
surface, but `train_target="gpu"`: the SAC sub-update + dynamics-ensemble
training + synthetic rollouts run on-device (the env is still stepped on
CPU — MBPO uses the single-env driver path). Because the GPU can afford the
legacy MBPO data regime, this example uses the LEGACY hyperparameters
(num_rollouts=100k, sac_updates=40, real_ratio=0.05, target_entropy=-3),
which is what gives the original `mbpo_half_cheetah_training_gpu.mojo` its
smooth convergence — and which measurably HURT on CPU (the CPU example keeps
a conservative regime; see its comments).

Carries the same convergence fixes as the CPU path:
  * Plain MLP critic (reference MBPO / rlkit SAC). An earlier LayerNorm critic
    suppressed value growth (Q stuck ~26, return plateaued ~250); dropping it
    let the return climb to ~3600 @ 80k with no divergence.
  * Dynamics input normalization (per-DYN_IN z-score, refit each model-train
    round) — essential for HalfCheetah's unbounded obs.
  * Elite ranking + holdout early-stop scored on plain MSE of the MEAN head
    (reference `inc_var_loss=False`), NOT the full NLL — NLL rewards over-
    confidence (shrinking variance), which makes synthetic rollouts near-
    deterministic and biased. Early-stop uses a 1%-RELATIVE improvement test.
  * `target_entropy=-6` (reference 'auto' = -ACT_DIM) and `sac_updates=20`
    (reference n_train_repeat). `REPLAY_CAPACITY >= NUM_STEPS` so the dynamics
    holdout split never leaks (see the constant's comment).

DynNet output layout: `2 * (1 + OBS_DIM)` = `[r_mean, r_logvar,
Δobs_mean[OBS_DIM], Δobs_logvar[OBS_DIM]]`. Logvar clamped to
`[LOGVAR_MIN, LOGVAR_MAX]`.

HalfCheetah (Physics3dEnv): 17D obs, 6D action, reward ≈ forward velocity −
0.1·||action||², no early termination.

NOTE: the GPU scaler-fit D2Hs the real buffer each model-train round; the
real buffer is capped at 200k (not the legacy's 1M) to keep that copy cheap.
A device-side reduction would let it grow — a future optimization.

Run:
    pixi run -e apple mojo run -I . examples/half_cheetah/mbpo_half_cheetah_nn_gpu.mojo    # Apple
    pixi run -e nvidia mojo run -I . examples/half_cheetah/mbpo_half_cheetah_nn_gpu.mojo   # NVIDIA
"""

from std.random import seed
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.io.artifact_sink import sink_for_run
from noeira.nn.constants import DT
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import ReLU
from noeira.nn.primitives.elementwise import Elementwise
from noeira.nn.primitives.ops.swish_op import SwishOp
from noeira.deep_agents.primitives.stochastic_actor import StochasticActor
from noeira.deep_agents.mbpo import MBPOAgent
from noeira.envs.half_cheetah import HalfCheetah, HalfCheetahConfig


# =============================================================================
# Architecture
# =============================================================================

comptime OBS_DIM = HalfCheetahConfig.OBS_DIM  # 17
comptime ACT_DIM = HalfCheetahConfig.ACTION_DIM  #  6
comptime HIDDEN = 256
comptime DYN_HIDDEN = 200
comptime BATCH = 128  # legacy MBPO batch size
# MUST be >= NUM_STEPS: the dynamics train/holdout split uses fixed physical
# slot ranges ([0,n_train) train, [n_train,n_data) holdout), which only stay
# disjoint while the real ring buffer has NOT wrapped. If it wraps, holdout
# leaks into training and elite/early-stop run on a contaminated signal. The
# scaler D2H copies the whole real buffer each model-train round, so this is
# the cost ceiling (300k×17 floats ≈ 20 MB — still cheap).
comptime REPLAY_CAPACITY = 300_000
comptime SYNTH_CAPACITY = 400_000
comptime N_ENSEMBLE = 7
comptime NUM_ELITES = 5
# Legacy GPU value: lean hard on the (now-normalized) world model. Affordable
# here ONLY because num_rollouts=100k keeps the synthetic buffer FRESH and
# in-distribution. Do NOT copy this to the CPU example.
comptime REAL_RATIO_PCT = 5
comptime LOGVAR_MIN_F = -10.0
# Kept at -5.0. The AdamW weight-decay fix 2.6×'d reward (337→873 @40k); the
# still-high holdout NLL (~400 vs legacy ~42) reflects a still-imperfect MEAN
# model, NOT a too-tight variance bound — legacy's effective max_logvar is ~-2
# (learnable, init +0.5, learned DOWN to [-1,-2] via 0.01 L2 penalty; bnn.py),
# and legacy explicitly found fixed +0.5 "too loose → synthetic rollouts drift
# far out of dist." nn's own A/B agrees (-2 scored 115 vs -5's 210). Raising
# the ceiling would let the model mask a bad mean with large variance and make
# rollouts noisier. The faithful fix is LEARNABLE bounds + L2 penalty, not a
# looser fixed ceiling.
comptime LOGVAR_MAX_F = -5.0

# CUDA-graph capture of the SAC sub-update loop + per-member dynamics-train
# step (NVIDIA only; NoAMP + uniform replay). Profiling showed the GPU run is
# launch-bound (cuLaunchKernelEx ~64% of wall) — capture collapses the SAC
# loop to one cuGraphLaunch/env-step (~2.5× wall in the profile). Set False if
# enabling bf16/PER.
comptime USE_TRAIN_CUDA_GRAPH = True

comptime NUM_STEPS = 300_000  # MBPO needs ~10× fewer real steps than SAC
comptime PRINT_EVERY = 10_000
comptime DIAG_EVERY = 5_000
comptime CHECKPOINT_EVERY = 50_000

# ─── A/B: entropy-temperature (alpha) ablation ───────────────────────────────
# The nn-MBPO vs legacy overlay showed nn's auto-tuned alpha equilibrates
# 2–4× BELOW legacy (0.035–0.086 vs ~0.12), correlating with ~4× slower mean_q
# growth + a climbing critic loss + a timid (low mean_abs_action) policy.
#   FIX_ALPHA = False → arm A: auto-tuned alpha (alpha_lr live, init 0.2).
#   FIX_ALPHA = True  → arm B: alpha PINNED at legacy's level (alpha_lr=0 so the
#                       ScalarAdam update is a no-op → alpha frozen at init).
# If arm B tracks legacy's mean_q / reward, alpha is confirmed as THE lever.
# Reverted to auto-α: the α A/B was REFUTED (fixed α=0.12 left reward flat at
# ~200 vs auto's ~210; the policy wasn't timid — mean_abs_action≈0.48). The
# real lever is the dynamics uncertainty bound (LOGVAR_MAX above).
comptime FIX_ALPHA = False
comptime FIXED_ALPHA: Scalar[DT] = 0.12  # legacy's stable equilibrium
comptime INIT_ALPHA: Scalar[DT] = FIXED_ALPHA if FIX_ALPHA else 0.2
comptime ALPHA_LR: Scalar[DT] = 0.0 if FIX_ALPHA else 3e-4
comptime RUN_NAME = (
    "MBPO HalfCheetah NN (GPU) — early-stop+elite, fixed alpha=0.12" if FIX_ALPHA else "MBPO HalfCheetah NN (GPU) — MSE-elite/holdout + target_entropy=-6, UTD=20"
)


comptime ActorNet = StochasticActor[
    OBS_DIM,
    ACT_DIM,
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
]
# Plain MLP critic — matches reference MBPO (rlkit SAC). An earlier LayerNorm
# critic (Q-explosion guard) suppressed value growth: Q saturated low (~26)
# and the return plateaued at ~250. Dropping it let the return climb cleanly
# (~3600 @ 80k env-steps) with no sign of divergence.
comptime CriticNet = Sequential[
    Linear[OBS_DIM + ACT_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]
# Dynamics output = 2 * (1 + OBS_DIM) = 2 * 18 = 36
# Layout: [r_mean, r_logvar, Δobs_mean[OBS_DIM], Δobs_logvar[OBS_DIM]]
comptime DynNet = Sequential[
    Linear[OBS_DIM + ACT_DIM, DYN_HIDDEN],
    Elementwise[DYN_HIDDEN, SwishOp],
    Linear[DYN_HIDDEN, DYN_HIDDEN],
    Elementwise[DYN_HIDDEN, SwishOp],
    Linear[DYN_HIDDEN, DYN_HIDDEN],
    Elementwise[DYN_HIDDEN, SwishOp],
    Linear[DYN_HIDDEN, DYN_HIDDEN],
    Elementwise[DYN_HIDDEN, SwishOp],
    Linear[DYN_HIDDEN, 2 * (1 + OBS_DIM)],
]


def main() raises:
    seed(42)
    print("=" * 70)
    print("MBPO (deep_agents) — HalfCheetah GPU (legacy hyperparams)")
    print("=" * 70)
    print("  OBS_DIM            =", OBS_DIM)
    print("  ACT_DIM            =", ACT_DIM)
    print("  HIDDEN (SAC)       =", HIDDEN)
    print("  DYN_HIDDEN         =", DYN_HIDDEN)
    print("  BATCH              =", BATCH)
    print("  REPLAY_CAPACITY    =", REPLAY_CAPACITY)
    print("  SYNTH_CAPACITY     =", SYNTH_CAPACITY)
    print("  N_ENSEMBLE/ELITES  =", N_ENSEMBLE, "/", NUM_ELITES)
    print("  REAL_RATIO_PCT     =", REAL_RATIO_PCT)
    print("  NUM_STEPS          =", NUM_STEPS)
    print("=" * 70)

    with DeviceContext() as ctx:
        # ─── Run + logger ───────────────────────────────────────────────────
        var run = RunContext(
            project=String("mujoco"),
            driver=String("examples/half_cheetah/mbpo_half_cheetah_nn_gpu.mojo"),
            slug=String("mbpo-half-cheetah-gpu"),
            env=String("builtin:mujoco/half_cheetah"),
        )
        var checkpoint_path = run.checkpoint_path(String("last"))
        print("  Run                =", run.dir)
        var logger = run_logger(run, buffer_size=64)
        logger.set_config("algorithm", "MBPO")
        logger.set_config("env", "HalfCheetah")
        logger.set_config("target", "gpu")
        logger.set_config("alpha_mode", "fixed_0.12" if FIX_ALPHA else "auto")
        logger.set_config("logvar_max", String(LOGVAR_MAX_F))
        logger.set_config("hidden", String(HIDDEN))
        logger.set_config("dyn_hidden", String(DYN_HIDDEN))
        logger.set_config("batch", String(BATCH))
        logger.set_config("ensemble", String(N_ENSEMBLE))
        logger.set_config("real_ratio_pct", String(REAL_RATIO_PCT))
        register_run(run, logger)
        var artifacts = sink_for_run(run.id, run.dir)

        var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()

        # ─── Agent + env ─────────────────────────────────────────────────
        var agent = MBPOAgent[
            "gpu",
            ActorNet,
            CriticNet,
            DynNet,
            OBS_DIM,
            ACT_DIM,
            BATCH,
            REPLAY_CAPACITY,
            SYNTH_CAPACITY,
            N_ENSEMBLE,
            NUM_ELITES,
            REAL_RATIO_PCT,
            LOGVAR_MIN_F,
            LOGVAR_MAX_F,
            USE_TRAIN_CUDA_GRAPH,
        ](
            ctx=ctx,
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=ALPHA_LR,  # A/B: 0.0 freezes alpha (arm B), 3e-4 = auto
            model_lr=1e-3,
            gamma=0.99,
            tau=0.005,
            action_scale=1.0,
            init_alpha=INIT_ALPHA,  # A/B: 0.12 (arm B) vs 0.2 (arm A)
            target_entropy=-6.0,  # reference MBPO: 'auto' = -ACT_DIM = -6
            learning_starts=5_000,  # legacy warmup
            window_size=100,
            initial_episode_fill=0.0,
            # Legacy GPU cadences — affordable on-device; the large fresh
            # synthetic buffer is what keeps the high-UTD critic stable.
            model_train_freq=250,
            dyn_epochs_per_round=4,
            rollout_length=1,
            num_rollouts_per_step=100_000,
            sac_updates_per_step=20,  # reference n_train_repeat (was 40)
            dyn_batch_size=256,
            # Ceiling on dyn-train epochs/round; early-stop on holdout NLL
            # governs in practice (matches legacy's 150 cap).
            dyn_max_epochs=150,
            # ROOT-CAUSE FIX #1: the nn dynamics ensemble used plain Adam (no
            # weight decay) → catastrophic overfit (train NLL → -19, holdout
            # NLL → 100+ vs legacy ~42) → optimistic OOD synthetic data that
            # adds ~nothing. Legacy uses AdamW with dyn_weight_decay=5e-5
            # (PETS/MBPO reference). Now matched (2.6× reward).
            dyn_weight_decay=5e-5,
            # ROOT-CAUSE FIX #2: the variance head was pinned over-confident by
            # the fixed LOGVAR_MAX, leaving holdout NLL ~400 vs legacy ~42.
            # Legacy uses LEARNABLE per-member/per-dim logvar bounds (soft
            # double-softplus clamp, init +0.5/−10, learned down to ~[−1,−2]
            # via a 0.01 L2 penalty). This is the last structural diff.
            dyn_learnable_bounds=True,
        )
        var env = HalfCheetah[DT, TERMINATE_ON_UNHEALTHY=False]()

        # ─── Train ───────────────────────────────────────────────────────
        var t_start = perf_counter_ns()
        _ = agent.train_single[
            HalfCheetah[DT, TERMINATE_ON_UNHEALTHY=False],
            L=RunLogger,
        ](
            env,
            NUM_STEPS,
            print_every=PRINT_EVERY,
            verbose=True,
            logger=logger_ptr,
            diag_every=DIAG_EVERY,
            checkpoint_path=checkpoint_path,
            checkpoint_every=CHECKPOINT_EVERY,
            artifacts=artifacts,
            run_dir=run.dir,
        )
        var elapsed_s = Float64(perf_counter_ns() - t_start) / 1e9
        var sent = logger.b.total_logged()
        finish_run(
            run, logger, artifacts,
            String("mean_return_100=") + String(agent.mean_return()),
        )
        _ = logger  # lifetime extender for logger_ptr

        # ─── Summary ─────────────────────────────────────────────────────
        print("=" * 70)
        print("Training complete")
        print("  total env_steps        =", NUM_STEPS)
        print("  elapsed                =", elapsed_s, "s")
        print("  mean ep return (last 100) =", agent.mean_return())
        print("  episodes completed     =", agent.ep_count())
        print("  remote points sent     =", sent)
        print("  run record             =", run.kv_path())
        print("=" * 70)

        var final_avg = Float64(agent.mean_return())
        if final_avg > 4000.0:
            print("EXCELLENT — running fast (mean > 4000).")
        elif final_avg > 1000.0:
            print("STRONG — learned locomotion (mean > 1000).")
        elif final_avg > 100.0:
            print("PROGRESS — early locomotion (mean > 100).")
        elif final_avg > 0.0:
            print("LEARNING — positive return (mean > 0).")
        else:
            print("EARLY — still exploring (mean < 0).")
        print("=" * 70)
