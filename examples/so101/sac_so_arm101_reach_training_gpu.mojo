"""SAC training on SO-ARM101 reach (GPU, multi-env).

The sim half of `ROADMAP_2026_08.md` §5.4's vertical: train here, evaluate on
CPU with `sac_so_arm101_reach_eval_cpu.mojo`, then run the same checkpoint on
the physical follower through `noeira/robot/so101/`.

Shaped exactly like `examples/half_cheetah/sac_half_cheetah_training_gpu.mojo`
— same `SACAgent["gpu", ...]` facade, same batched off-policy driver, same
one-file `nn-ckpt v2` output — so the two are diffable and a change to the
driver shows up in both. The run (project `so101`) keeps its checkpoint
(`runs/<id>/checkpoints/last.ckpt`), `metrics.csv` and `run.kv` together.

⚠⚠ **THE ACTION SPACE IS [-1, 1] PER JOINT**, mapped affinely onto each
joint's own `ctrlrange` by the env (`SoArmReachConfig.NORMALIZED_ACTIONS`). So
`ACTION_SCALE = 1.0`, and every other script that builds an agent for this env
must use 1.0 too.

⚠⚠ IT USED TO BE RADIANS WITH `ACTION_SCALE = 2.0`, and that was the defect
behind the shaking. One scalar scale cannot fit six joints whose ranges run
1.66 to 2.84: at 2.0 the tanh rails sat OUTSIDE most ranges, so the trained
policy commanded out-of-range poses on **24% to 100% of control steps**,
`elbow_flex` was railed 49% of the time, and the gripper — asymmetric
-0.17..1.75 against a symmetric +-2.0 — was out of range on EVERY step. Past
the clamp the gradient is ZERO: a whole band of actor outputs maps to one
pose, so the actor drifts across it for free and flips to the far rail for
free. Three successive reward shapes produced the same shaking arm because the
clamp was eating the signal each of them tried to send.

⚠ The asymmetric gripper is now handled correctly rather than being 100%
clamped — the affine map is per-joint and uses each range's true endpoints.

Task (`SoArmReachConfig`): a mocap target drawn per episode from an azimuth
cone × elevation band × radial shell (**0.18–0.30 m** — the near end was raised
from 0.15 after measurement showed 1/6 reached below 0.17 against 14/18 above);
reward is dm_control's shaped `tolerance` on jaw-to-target distance with a 2 cm
radius and a **0.05 m** margin, times a stillness term on joint speed; no early
termination.

  * 21D observation — qpos(6) + qvel(6) + ee(3) + target(3) + ee_to_target(3)
  * 6D continuous action — joint angle targets, radians

Run:
    pixi run -e nvidia mojo run -I . examples/so101/sac_so_arm101_reach_training_gpu.mojo
    ... --resume --ckpt runs/<id>/checkpoints/last.ckpt   # fine-tune
    ... --resume --ckpt <ckpt> --steps 100000              # a shorter one
    ... --resume --ckpt <ckpt> --alpha 0.02

⚠⚠ **`--resume` IS THE RIGHT WAY TO PICK UP `REWARD_MARGIN = 0.05`.** The
margin was 0.25 — twelve times the success radius — and at 0.05 a target
100 mm away is worth 0.003 per step, so a policy that cannot already find the
target learns nothing from scratch. Exploration is demonstrably solved (the
current checkpoint reaches 3.9 mm on real hardware); the tighter margin only
sharpens the endgame, which is exactly what a fine-tune is for.
"""

from max.gpu.host import DeviceContext
from std.random import seed
from std.sys import argv
from std.time import perf_counter_ns

from noeira.core.run import RunContext, register_run, run_id_of_checkpoint
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.io.artifact_sink import sink_for_run
from noeira.nn.constants import DT
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.linear_relu import LinearReLU
from noeira.deep_agents.primitives.stochastic_actor import StochasticActor
from noeira.deep_agents.sac import SACAgent
from noeira.deep_agents.training.blocks import UniformSampleGpuStep
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.envs.robots.so_arm101_xml import SoArm101Model
from noeira.envs.robots.so_arm101 import SoArm101ReachConfig


comptime N_ENVS = 32

comptime BatchedEnvT = Phyics3dBatchedEnv[
    SoArm101Model, SoArm101ReachConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=False
]

comptime OBS_DIM = BatchedEnvT.OBS_DIM  # 27 (incl. the previous action)
comptime ACT_DIM = 6
comptime HIDDEN = 256

comptime BATCH = 256
comptime REPLAY_CAPACITY = 1_000_000

comptime NUM_STEPS = 400_000
comptime WARMUP_STEPS = 10_000

# ── fine-tuning (`--resume`) ──────────────────────────────────────────────
#
# ⚠⚠ `learning_starts` GATES ACTING, NOT JUST UPDATES. `SACTrainer.
# select_action_batched` opens with
#
#     if step_idx < self.learning_starts:
#         warmup_uniform_batched[...]        # uniform in [-scale, +scale]
#
# so a resume that kept `WARMUP_STEPS = 10_000` would load a trained policy
# and then take TEN THOUSAND ENV-STEPS OF UNIFORM RANDOM ACTIONS, fill the
# replay with them, and begin updating the loaded weights against that. The
# checkpoint would be damaged before the first useful gradient. A fine-tune
# needs only enough prefill to fill one batch — 1000 env-steps is 1000
# transitions against a BATCH of 256, with the LOADED policy generating them.
comptime RESUME_WARMUP = 1_000
# ⚠ AND THE CHECKPOINT DOES NOT CARRY ALPHA. `SACAgent.save` writes actor +
# both online critics and nothing else — "optimizer moments, alpha, replay
# buffer and episode tracker are NOT included; resume re-warms". So a resume
# restarts the entropy temperature at `init_alpha`, and 0.2 on an already
# competent policy is a lot of exploration noise poured over it. 0.05 keeps
# some exploration — which this fine-tune WANTS, since the behaviour it has to
# find (arrive and hold still) is not the one it currently has — without
# washing the policy out. `--alpha` overrides.
comptime RESUME_ALPHA = Scalar[DT](0.05)
comptime PRINT_EVERY = 25_000
comptime DIAG_EVERY = 1000
comptime CHECKPOINT_EVERY = 25_000

# See the module docstring: radians, not a unit box.
comptime ACTION_SCALE = Scalar[DT](1.0)

comptime ActorNet = StochasticActor[
    OBS_DIM,
    ACT_DIM,
    LinearReLU[OBS_DIM, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
]
comptime CriticNet = Sequential[
    LinearReLU[OBS_DIM + ACT_DIM, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
    Linear[HIDDEN, 1],
]


def main() raises:
    seed(42)

    # ── flags ─────────────────────────────────────────────────────────────
    #   --resume        fine-tune from --ckpt instead of scratch
    #   --ckpt PATH     the checkpoint --resume reads (writes always go to
    #                   the run's own `checkpoints/last.ckpt`)
    #   --alpha X       initial entropy temperature (see RESUME_ALPHA)
    #   --steps N       env-steps to run (a fine-tune wants far fewer)
    var resume = False
    var ckpt_path = String("")
    var num_steps = NUM_STEPS
    var init_alpha = Scalar[DT](0.2)
    var alpha_set = False
    var args = argv()
    for i in range(1, len(args)):
        var a = String(args[i])
        if a == "--resume":
            resume = True
        elif a == "--ckpt" and i + 1 < len(args):
            ckpt_path = String(args[i + 1])
        elif a == "--steps" and i + 1 < len(args):
            num_steps = Int(String(args[i + 1]))
        elif a == "--alpha" and i + 1 < len(args):
            init_alpha = Scalar[DT](Float64(String(args[i + 1])))
            alpha_set = True
    if resume and ckpt_path.byte_length() == 0:
        print("ERROR: --resume needs --ckpt PATH (e.g. a previous run's")
        print("runs/<id>/checkpoints/last.ckpt).")
        return
    if resume and not alpha_set:
        init_alpha = RESUME_ALPHA
    var warmup = RESUME_WARMUP if resume else WARMUP_STEPS

    print("=" * 70)
    if resume:
        print("SAC — SO-ARM101 reach, GPU — FINE-TUNE from a checkpoint")
    else:
        print("SAC — SO-ARM101 reach, GPU (multi-env)")
    print("=" * 70)
    print("  resume           =", resume, "(" + ckpt_path + ")")
    print("  warmup steps     =", warmup, "(RANDOM actions until then)")
    print("  init_alpha       =", init_alpha)
    print("  OBS_DIM          =", OBS_DIM)
    print("  ACT_DIM          =", ACT_DIM, "(joint angles in RADIANS)")
    print("  HIDDEN           =", HIDDEN)
    print("  N_ENVS           =", N_ENVS)
    print("  NUM_STEPS        =", num_steps)
    print("  action_scale     =", ACTION_SCALE)

    var run = RunContext(
        project=String("so101"),
        driver=String("examples/so101/sac_so_arm101_reach_training_gpu.mojo"),
        slug=String("sac-so101-reach"),
        env=String("builtin:so_arm101/reach"),
        seed=42,
        resumed_from=(
            run_id_of_checkpoint(ckpt_path) if resume else String("")
        ),
    )
    # `project-resume <run_id>` continues this run from its last checkpoint.
    # ⚠ This driver's `--resume` re-warms and resets alpha (see the header):
    # it is a continuation, not a bit-exact resume.
    run.set_resume_args(String("--resume --ckpt {ckpt}"))
    var out_ckpt = run.checkpoint_path(String("last"))
    print("  run              =", run.dir)
    print("=" * 70)

    with DeviceContext() as ctx:
        var logger = run_logger(run, buffer_size=64)
        logger.set_config("algorithm", "SAC")
        logger.set_config("env", "SoArm101Reach")
        logger.set_config("target", "gpu")
        logger.set_config("n_envs", String(N_ENVS))
        register_run(run, logger)
        var artifacts = sink_for_run(run.id, run.dir)
        var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()

        var agent = SACAgent[
            "gpu",
            UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
            ActorNet,
            CriticNet,
        ](
            ctx=ctx,
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=3e-4,
            gamma=0.99,
            tau=0.005,
            action_scale=ACTION_SCALE,
            init_alpha=init_alpha,
            target_entropy=-Scalar[DT](ACT_DIM),
            learning_starts=warmup,
            window_size=100,
            # ⚠⚠ NOT THE DEFAULT. SAC seeds its return window with
            # `initial_episode_fill = -1250.0`, a HalfCheetah-flavoured
            # pessimistic value. This task's reward is a shaped `tolerance` in
            # [0, 1], so a return is in [0, 500] and CANNOT be negative — with
            # the default, every reading before the window fills is a blend of
            # real returns and sentinels, and it reads as a broken reward.
            #
            # Measured: a 24k-step smoke run printed `mean_ret = -793.2661`
            # at 32 episodes. Solving (32R + 68*(-1250))/100 for R gives
            # R = 177.3, reported as "still exploring". The template sets this
            # to 0.0 for the same reason; dropping the line is what produced
            # the false reading.
            #
            # ⚠ AND 177 IS NOT YET "REACHING TARGETS", which an earlier version
            # of this comment claimed. The untrained baseline is 46 (see the
            # verdict bands below), so 177 is ~4x the floor after 24k steps —
            # real movement, not a solved task.
            initial_episode_fill=0.0,
        )
        var env = BatchedEnvT(ctx)

        # ⚠⚠ A FAILED `--resume` MUST NOT FALL BACK TO TRAINING FROM SCRATCH.
        # The two runs look identical in the log until the return curve starts
        # from zero hours later, and the wasted run is the whole cost of the
        # mistake. Refuse instead.
        if resume:
            try:
                agent.load(ckpt_path)
                print("Resumed from", ckpt_path)
                print(
                    "  ⚠ actor + both online critics restored; optimizer"
                    " moments, alpha and the\n    replay buffer are NOT in the"
                    " envelope and re-warm from scratch."
                )
            except e:
                print("ERROR: --resume given but", ckpt_path, "did not load:")
                print("   ", e)
                print("Refusing to silently train from scratch. Drop --resume")
                print("to start fresh, or pass --ckpt with the right path.")
                return

        print("Starting GPU training...")
        print("-" * 70)
        var t_start = perf_counter_ns()
        _ = agent.train[
            BatchedEnvT,
            N_ENVS=N_ENVS,
            USE_TRAIN_CUDA_GRAPH=True,
            USE_ENV_CUDA_GRAPH=False,
            L=RunLogger,
        ](
            env,
            num_steps,
            rng_seed=UInt64(42),
            updates_per_step=N_ENVS,
            print_every=PRINT_EVERY,
            verbose=True,
            logger=logger_ptr,
            diag_every=DIAG_EVERY,
            episode_sync_every=32,
            checkpoint_every=CHECKPOINT_EVERY,
            checkpoint_path=out_ckpt,
            artifacts=artifacts,
            run_dir=run.dir,
        )
        var elapsed_s = Float64(perf_counter_ns() - t_start) / 1e9
        finish_run(
            run, logger, artifacts,
            String("mean_return_100=") + String(agent.mean_return()),
        )
        _ = logger

        print("-" * 70)
        print("Training complete")
        print("  total env_steps           =", num_steps)
        print("  elapsed                   =", elapsed_s, "s")
        print("  mean ep return (last 100) =", agent.mean_return())
        print("  episodes completed        =", agent.ep_count())
        print("  checkpoint                =", out_ckpt)
        print("=" * 70)

        # ⚠ Reward is a shaped `tolerance` in [0, 1] per step over
        # `MAX_STEPS = 500` control steps, so the ceiling is 500 and a policy
        # that reaches the target quickly and HOLDS it is what scores.
        #
        # ⚠⚠ THE FLOOR IS NOT ZERO, AND THAT IS THE WHOLE POINT OF THESE
        # BANDS. An UNTRAINED actor measured **45.9** over 11 episodes on this
        # env (greedy, 2026-08-26; per-episode 3.5 .. 123.8) — the 0.25 m
        # reward margin covers most of a 0.15-0.30 m target shell, so flailing
        # near the middle of the workspace pays. Anything under ~90 has not
        # been shown to have learned anything. Same bands as
        # `sac_so_arm101_reach_eval_cpu.mojo`; keep them in step.
        var m = Float64(agent.mean_return())
        if m > 400.0:
            print("EXCELLENT — reaches and holds (mean > 400 / 500).")
        elif m > 200.0:
            print("STRONG — reaches most targets (mean > 200).")
        elif m > 90.0:
            print("PROGRESS — measurably above the untrained baseline (~46).")
        else:
            print("NO BETTER THAN AN UNTRAINED NET (baseline mean ~46).")
        print("=" * 70)
