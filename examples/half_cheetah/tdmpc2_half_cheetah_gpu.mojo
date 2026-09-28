"""TD-MPC2 (deep_agents) — HalfCheetah training (GPU) via the `agent.train` facade.

Mirrors `examples/humanoid/sac_humanoid_training_gpu.mojo`: build the agent
through the `TDMPC2[...]` preset (config.mojo, reference-tuned defaults), then
run a SINGLE `agent.train[...](env, TOTAL, ...)` call. The driver internalizes
the collect → record → train_step loop plus warmup, periodic deterministic eval
(isolated env), logging, and checkpointing — no hand-rolled loop here.

Built for the NVIDIA run (`pixi run -e nvidia`): on CUDA the GPU path is fast
(low per-launch overhead + grouped multi-tensor Adam + big-matmul-dominated),
whereas on Apple/Metal TD-MPC2 is kernel-launch-bound and CPU is faster (see
tests/deep_agents/test_tdmpc2_perf.mojo).

This script uses the SINGLE-env driver. `agent.train_batched[E: BatchedEnv,
N_ENVS]` also exists now (added 2026-08-11): the MPPI planner has always been
N_ENVS-batched, and the world-model update never depended on how the data was
collected — see `examples/dm_control/tdmpc2_dm_walker_batched_gpu.mojo`.

Acting mode (comptime `USE_MPC`):
  * False → `a = π(encode(obs))` (MPC-off, fast).
  * True  → `a = MPPI plan` over the world model (`select_action_mpc`, GPU
    only). Much heavier per env step; `MPC_*` set the planning budget
    (reference TD-MPC2 is 512/24/64/6 — start lighter for a feasible run).

HalfCheetah (Phyics3dEnv, MuJoCo-style):
  * 17D obs, 6D action (joint torques in [-1,1]).
  * No early termination — truncates at the horizon, so the driver records
    `done=0` throughout (the value bootstrap continues across truncation).
  * Reward ≈ forward velocity − control cost; good policies reach a few
    thousand return / 1000-step episode.

Dims follow the reference (latent 512, mlp 512, enc 256, num_bins 101,
num_q 5, horizon 3). Drop TOTAL / dims for a quick smoke.

Run:
    pixi run -e nvidia mojo run -I . examples/half_cheetah/tdmpc2_half_cheetah_gpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.io.artifact_sink import sink_for_run
from noeira.deep_agents.tdmpc2.config import TDMPC2
from noeira.envs.half_cheetah import HalfCheetah, HalfCheetahConfig

# ── target: "gpu" for the NVIDIA run; "cpu" works too (slower at this scale).
comptime TARGET = "gpu"
# ── MPC: True → act via MPPI planning (GPU only, heavy). MPC_* set the budget;
#    reference TD-MPC2 is 512/24/64/6 — start lighter for a feasible first run.
comptime USE_MPC = True
comptime MPC_SAMPLES = 256
comptime MPC_PI_TRAJS = 12
comptime MPC_ELITES = 32
comptime MPC_ITERS = 4

comptime OBS = HalfCheetahConfig.OBS_DIM        # 17
comptime ACT = HalfCheetahConfig.ACTION_DIM     #  6
comptime ENC = 256
comptime LATENT = 512
comptime MLP = 512
comptime BINS = 101
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
comptime B = 256
comptime H = 3
comptime CAP = 1_000_000

comptime LR = 3e-4
comptime ACTION_SCALE = 1.0
comptime LEARN_START = 5_000
comptime TRAIN_EVERY = 1
comptime TOTAL = 1_000_000
comptime EVAL_EVERY = 20_000
comptime EVAL_EPS = 2
comptime EP_LEN = 1_000
comptime DIAG_EVERY = 1_000   # metric-bundle flush → logger cadence
comptime PRINT_EVERY = 20_000
comptime CHECKPOINT_EVERY = 50_000

comptime Env = HalfCheetah[DT, TERMINATE_ON_UNHEALTHY=False]


def main() raises:
    print("=" * 70)
    var mode = "MPC" if USE_MPC else "MPC-off"
    print("TD-MPC2 (deep_agents) — HalfCheetah", TARGET, "(", mode, ")")
    print("  OBS=", OBS, " ACT=", ACT, " latent=", LATENT, " B=", B, " H=", H)
    print("  lr=", LR, " total=", TOTAL, " learn_start=", LEARN_START)
    print("=" * 70)
    seed(0)
    var ctx = DeviceContext()

    # Two CPU-stepped HalfCheetah envs: one for collection, one isolated for
    # deterministic eval (so eval resets never disturb the training rollout).
    var env = Env()
    var eval_env = Env()
    # `.as_unsafe_any_origin()` — the facade takes
    # Optional[Pointer[EE, MutAnyOrigin]]; a tracked-origin pointer
    # doesn't convert (same idiom as logger_ptr below).
    var eval_env_ptr = Pointer(to=eval_env).as_unsafe_any_origin()

    # Build through the Design-F preset (config.mojo): reads like a constructor,
    # applies the reference-tuned defaults (gamma 0.99 / tau 0.01 /
    # enc_lr_scale 0.3 / …), returns exactly the TDMPC2Agent we train below.
    var ag = TDMPC2[
        TARGET, OBS, ACT, B, CAP, ENC, LATENT, MLP, BINS, SN, VMIN, VMAX, H,
        MPC_SAMPLES, MPC_PI_TRAJS, MPC_ELITES, MPC_ITERS,
    ](
        ctx=ctx, lr=Scalar[DT](LR),
        action_scale=Scalar[DT](ACTION_SCALE), learning_starts=LEARN_START,
    )

    # ─── Run + logger ───────────────────────────────────────────────────────
    # The run directory replaces the old mode-specific checkpoint name: an MPC
    # run and an MPC-off run never share a directory (the mode is in the slug).
    var run = RunContext(
        project=String("mujoco"),
        driver=String("examples/half_cheetah/tdmpc2_half_cheetah_gpu.mojo"),
        slug=String("tdmpc2-half-cheetah-mpc") if USE_MPC else String(
            "tdmpc2-half-cheetah-mpcoff"
        ),
        env=String("builtin:mujoco/half_cheetah"),
    )
    var checkpoint_path = run.checkpoint_path(String("last"))
    print("  Run:", run.dir)
    var logger = run_logger(run, buffer_size=64)
    logger.set_config("algorithm", "TD-MPC2")
    logger.set_config("env", "HalfCheetah")
    logger.set_config("mpc", String("1") if USE_MPC else String("0"))
    register_run(run, logger)
    var artifacts = sink_for_run(run.id, run.dir)
    var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()

    # ─── Single train() call — single-env TD-MPC2 driver ─────────────────
    print("Starting training...")
    print("-" * 70)
    var t_start = perf_counter_ns()
    var best = ag.train[Env, RunLogger, Env, USE_MPC](
        env,
        TOTAL,
        train_every=TRAIN_EVERY,
        print_every=PRINT_EVERY,
        verbose=True,
        logger=logger_ptr,
        diag_every=DIAG_EVERY,
        checkpoint_path=checkpoint_path,
        checkpoint_every=CHECKPOINT_EVERY,
        artifacts=artifacts,
        run_dir=run.dir,
        eval_env=eval_env_ptr,
        eval_every=EVAL_EVERY,
        eval_episodes=EVAL_EPS,
        eval_max_steps=EP_LEN,
    )
    _ = eval_env  # lifetime extender for eval_env_ptr
    var elapsed = Float64(perf_counter_ns() - t_start) / 1e9

    finish_run(
        run, logger, artifacts, String("best_eval_return=") + String(best)
    )
    _ = logger  # lifetime extender for logger_ptr
    print("=" * 70)
    print("  FINAL best eval return =", best, " (", elapsed, "s )")
    print("  ( HalfCheetah: >3000 good, >8000 strong )")
    print("  run:", run.dir)
    print("=" * 70)
