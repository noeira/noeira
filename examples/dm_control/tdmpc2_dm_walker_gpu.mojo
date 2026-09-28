"""TD-MPC2 on dm_control `walker` — SINGLE task (GPU nets, CPU-stepped env).

This is step 1 of the multi-task plan: before asking TD-MPC2 to learn three
walker tasks under one task-conditioned world model
(`deep_agents/tdmpc2/agent_mt.mojo` + `TDMPC2MultiTask`), re-establish that the
SINGLE-task agent still learns on a task we already have a reference curve for.
`examples/dm_control/sac_dm_walker_training_gpu.mojo` trained SAC on the same
body and the same three rewards, so this script's return is directly comparable
to that run's — same env, same reward, same ceiling.

Why it needs re-validating at all: the last TD-MPC2 convergence evidence
predates the rc2 migration and the physics3d fields migration. The compile
smokes (`tests/deep_agents/test_tdmpc2_*`) prove the graphs still build and
`test_tdmpc2_pendulum_gpu_convergence` proves the GPU path still learns a
1-DOF task; neither proves it learns a 24-D contact-rich one.

## Structural difference from the SAC script

SAC there runs `Phyics3dBatchedEnv` with N_ENVS=32 on the GPU. This script uses
the SINGLE-env CPU facade `Phyics3dEnv` and the single-env `agent.train(...)`
driver: the nets and planner run on the GPU (`TARGET="gpu"`), the physics does
not. Expect a few env-steps/s, not SAC's thousands — `TOTAL` below is sized for
that.

⚠ For throughput, prefer the BATCHED sibling:
`examples/dm_control/tdmpc2_dm_walker_batched_gpu.mojo` runs N walkers in one
`Phyics3dBatchedEnv` through `agent.train_batched`. This single-env script is
now the REFERENCE/DEBUG path — the one whose collection order matches the
original TD-MPC2 loop one env-step at a time — and is the right thing to
compare a suspicious batched curve against.

Consequence for checkpoints: the driver OVERWRITES `checkpoint_path` on every
save, so this run leaves ONE file (`runs/<id>/checkpoints/last.ckpt`), not a
ladder. That is fine here — the FB ladder comes from the SAC script; this run
only needs a resumable final policy.

## Acting mode (`USE_MPC`)

  * True  → `a = MPPI plan` over the learned world model (`select_action_mpc`,
    GPU only). This is TD-MPC2 as published, and the thing being validated.
  * False → `a = π(encode(obs))`, much faster per step, but it validates only
    the model-free half of the algorithm.

`MPC_*` set the planning budget. The reference is 512/24/64/6 per step; the
defaults below (256/12/32/4) are the same lighter budget
`examples/half_cheetah/tdmpc2_half_cheetah_gpu.mojo` uses, because the budget
multiplies the per-step cost and the CPU env is already the bottleneck.

## Tasks

    stand  MOVE_SPEED = 0.0    walk  MOVE_SPEED = 1.0    run  MOVE_SPEED = 8.0

Edit `TASK`, rebuild, run — only the selected branch is instantiated. Start
with `walk`: `stand` is too easy to discriminate a broken agent (a frozen
policy already scores a few hundred) and `run` is the hardest of the three.

## Reading the result

dm_control rewards are in [0, 1] per step over 1000 steps, so an episode return
is bounded by 1000 REGARDLESS of task — unlike HalfCheetah's unbounded forward
velocity. Judge against that ceiling, and against the SAC run on the same task.
A run that is still under ~100 once training has had a few hundred thousand
steps is not "slow", it is broken — check `wm=` (world-model loss) in the
progress line first: it should fall and stay finite.

## Measured throughput (do not run the full thing on Apple)

Smoke run 2026-08-11, Apple/Metal, `USE_MPC=True`, B=64, MPPI 256/12/32/4:
**~2.9 env-steps/s** after warmup (800 steps in 273 s). At that rate `TOTAL`
below is ~29 h on Apple. This is an NVIDIA-first script — Apple is for the
smoke, NVIDIA for the run. (TD-MPC2 on Metal is kernel-launch-bound and the
"cpu" target is faster at this scale, `tests/deep_agents/test_tdmpc2_perf.mojo`
— but `USE_MPC=True` asserts `target == "gpu"`, so an Apple MPC run must stay
on "gpu" regardless.)

Cheap re-validation before committing to the long run: copy this file, set
TOTAL=1_200 / LEARN_START=200 / EP_LEN=100 / B=64, and check that `wm=` FALLS
(it went 0.96 → 0.11 over 1000 steps in the smoke). That exercises every piece
— env, replay, world-model BPTT, policy update, MPPI plan, eval, logger.

Run:
    pixi run -e nvidia mojo run -I . examples/dm_control/tdmpc2_dm_walker_gpu.mojo
    pixi run -e apple  mojo run -I . examples/dm_control/tdmpc2_dm_walker_gpu.mojo

⚠ `mojo build` of this file fails at LINK time on macOS (Apple `ld` asserts on
an over-long mangled name, `SymbolString.cpp:74`) — that is not a source error;
`mojo run` JITs past it, and `-Xlinker -ld_classic` fixes the build.

⚠ The GPU MPPI path did NOT compile under Mojo 1.0.0rc2 until 2026-08-11:
`mppi_sample_actions_batched_kernel` took its horizon-step index as `Int`,
which stopped conforming to `DevicePassable`. Nothing instantiated that path in
the package build, so it was dark; `tests/deep_agents/test_tdmpc2_agent_mpc_gpu.mojo`
is the gate that catches it and it now passes.
"""

from std.random import seed
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.io.artifact_sink import sink_for_run
from noeira.deep_agents.tdmpc2.config import TDMPC2
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.envs.dm_control.walker.walker_xml import DMWalkerModel
from noeira.envs.dm_control.walker.walker_config import DMWalkerConfig


# ── pick ONE ─────────────────────────────────────────────────────────────
comptime TASK: StaticString = "walk"  # "stand" | "walk" | "run"

comptime MOVE_SPEED: Float64 = 0.0 if TASK == "stand" else (
    1.0 if TASK == "walk" else 8.0
)

# ── target: "gpu" for the NVIDIA run; "cpu" works too (MPC-off only).
comptime TARGET = "gpu"
# ── MPC: True → act via MPPI planning (GPU only, heavy).
comptime USE_MPC = True
comptime MPC_SAMPLES = 256
comptime MPC_PI_TRAJS = 12
comptime MPC_ELITES = 32
comptime MPC_ITERS = 4

comptime OBS = DMWalkerModel.OBS_DIM       # 24 = 7 bodies x 2 + height + nv(9)
comptime ACT = DMWalkerModel.ACTION_DIM    #  6 hip/knee/ankle x 2
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
# Sized for the single-env CPU-stepped rate, not SAC's batched throughput.
# The SAC comparison run is 640 k env-steps over 32 envs; this is 300 k over 1.
comptime TOTAL = 300_000
comptime EVAL_EVERY = 10_000
comptime EVAL_EPS = 2
# dm_control's own limit: time_limit 25 s / control_timestep 0.025 s.
comptime EP_LEN = 1_000
comptime DIAG_EVERY = 1_000      # metric-bundle flush → logger cadence
comptime PRINT_EVERY = 5_000
comptime CHECKPOINT_EVERY = 25_000

# dm_control per-step reward is in [0, 1] over 1000 steps.
comptime MAX_RETURN = 1000.0

# `TERMINATE_ON_UNHEALTHY=False`: dm_control tasks never terminate early, so
# the driver only ever sees truncation at 1000 steps and records done=0
# throughout (the value bootstrap continues across the truncation).
#
# float64 physics: this is the dtype the dm_control parity tests validated the
# walker CPU path in. The driver converts obs/actions to `DT` (float32) at the
# net boundary anyway, so the choice only affects the physics itself.
comptime Env = Phyics3dEnv[
    DMWalkerModel, DMWalkerConfig[MOVE_SPEED], DType.float64, False
]


def main() raises:
    # ⚠ In a function body — a top-level `comptime assert` does not parse.
    # Without it a typo'd TASK falls through the ternary above to 8.0 (run)
    # and silently trains the wrong task under the right filename.
    comptime assert (
        TASK == "stand" or TASK == "walk" or TASK == "run"
    ), "TASK must be 'stand', 'walk' or 'run'"

    var mode = "MPC" if USE_MPC else "MPC-off"
    print("=" * 70)
    print("TD-MPC2 — dm_control walker", TASK, "(", TARGET, "/", mode, ")")
    print("=" * 70)
    print("  TASK / MOVE_SPEED =", TASK, "/", MOVE_SPEED)
    print("  OBS =", OBS, " ACT =", ACT, " latent =", LATENT, " B =", B,
          " H =", H)
    print("  lr =", LR, " total =", TOTAL, " learn_start =", LEARN_START)
    comptime if USE_MPC:
        print("  MPPI budget =", MPC_SAMPLES, "samples /", MPC_PI_TRAJS,
              "pi-trajs /", MPC_ELITES, "elites /", MPC_ITERS, "iters")
    print("=" * 70)
    seed(0)
    var ctx = DeviceContext()

    # Two CPU-stepped walkers: one for collection, one isolated for
    # deterministic eval (so eval resets never disturb the training rollout).
    var env = Env()
    var eval_env = Env()
    # `.as_unsafe_any_origin()` — the facade takes
    # Optional[Pointer[EE, MutAnyOrigin]]; a tracked-origin pointer doesn't
    # convert (same idiom as logger_ptr below).
    var eval_env_ptr = Pointer(to=eval_env).as_unsafe_any_origin()

    # One directory per run; the mode and task are in the slug so an MPC run
    # and an MPC-off one, or `walk` and `run`, are told apart by name.
    var run = RunContext(
        project=String("dm-control"),
        driver=String("examples/dm_control/tdmpc2_dm_walker_gpu.mojo"),
        slug=(
            String("tdmpc2-dm-walker-") + String(TASK)
            + ("-mpc" if USE_MPC else "-mpcoff")
        ),
        env=String("builtin:dm_control/walker-") + String(TASK),
    )
    var checkpoint_path = run.checkpoint_path(String("last"))
    print("  Run:", run.dir)

    # Build through the Design-F preset (config.mojo): reads like a
    # constructor, applies the reference-tuned defaults (gamma 0.99 / tau 0.01
    # / enc_lr_scale 0.3 / …), returns exactly the TDMPC2Agent we train below.
    var ag = TDMPC2[
        TARGET, OBS, ACT, B, CAP, ENC, LATENT, MLP, BINS, SN, VMIN, VMAX, H,
        MPC_SAMPLES, MPC_PI_TRAJS, MPC_ELITES, MPC_ITERS,
    ](
        ctx=ctx, lr=Scalar[DT](LR),
        action_scale=Scalar[DT](ACTION_SCALE), learning_starts=LEARN_START,
    )

    # metrics.csv + the monitor from .env (the remote half no-ops if unset).
    var logger = run_logger(run, buffer_size=64)
    logger.set_config("algorithm", "TD-MPC2")
    logger.set_config("env", String("dm_control/walker-") + String(TASK))
    logger.set_config("target", TARGET)
    logger.set_config("mpc", String("1") if USE_MPC else String("0"))
    logger.set_config("latent", String(LATENT))
    logger.set_config("batch", String(B))
    logger.set_config("horizon", String(H))
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

    print("-" * 70)
    print("=" * 70)
    print("Training complete —", TASK, "(", mode, ")")
    print("  total env_steps        =", TOTAL)
    print("  elapsed                =", elapsed, "s")
    print("  best eval return       =", best)
    print("  run                    =", run.dir)
    print("=" * 70)

    var frac = Float64(best) / MAX_RETURN
    if frac > 0.8:
        print("EXCELLENT — near the dm_control ceiling (>0.8 x 1000).")
    elif frac > 0.5:
        print("STRONG — solved the task (>0.5 x 1000).")
    elif frac > 0.2:
        print("PROGRESS — partial competence (>0.2 x 1000).")
    else:
        print("WEAK — TD-MPC2 did not learn this task. Check `wm=` in the")
        print("  progress lines: a loss that is flat or non-finite means the")
        print("  world model, not the budget, is the problem.")
    print("")
    print("Compare against SAC on the SAME task (same env, same reward):")
    print("  examples/dm_control/sac_dm_walker_training_gpu.mojo")
    print("")
    print("Next (once this is convincing): multi-task over stand/walk/run via")
    print("  TDMPC2MultiTask[...] (deep_agents/tdmpc2/config_mt.mojo) — one")
    print("  task-conditioned world model, obs/act padded to MAX_OBS/MAX_ACT.")
    print("  ⚠ the MT agent has no `train` driver yet; that loop is the next")
    print("  script to write.")
    print("=" * 70)
