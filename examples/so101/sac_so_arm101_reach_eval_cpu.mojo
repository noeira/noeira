"""Evaluate + render a trained SO-ARM101 reach checkpoint (CPU, single env).

The middle rung of `ROADMAP_2026_08.md` §5.4's vertical: train GPU-batched with
`sac_so_arm101_reach_training_gpu.mojo`, watch it here, then run the SAME
checkpoint on the physical follower with `deploy_reach_real.mojo`.

Counterpart of `examples/half_cheetah/sac_half_cheetah_nn_eval_cpu.mojo`, and
built through the same `SAC[...]` preset so the checkpoint's parameter layout
matches the trainer's exactly.

⚠⚠ `action_scale` MUST match training. The greedy action is
`a = tanh(mu) * action_scale`, and for these arms `a` is a JOINT ANGLE IN
RADIANS, not a normalised torque — so a mismatched scale does not merely
weaken the policy, it commands a different pose. `2.0` is the trainer's value.

⚠ THE OBSERVATION MUST MATCH TOO, and that is a sharper constraint than it
looks: the trainer fills it with `custom_extract_obs_gpu` and this eval with
`custom_extract_obs_cpu`. Those are two implementations of one contract —
qpos(6) + qvel(6) + ee(3) + target(3) + ee_to_target(3) — and a permutation in
either is a policy that works on one device and is nonsense on the other, with
no error anywhere.

⚠ RESET FOLDS THE ARM TO HOME and draws a fresh target each episode; the
target is a mocap body, so it is visible in the render as the thing the jaw
should be touching.

Run:
    pixi run mojo run -I . examples/so101/sac_so_arm101_reach_eval_cpu.mojo --ckpt <run_id>
"""

from std.random import seed
from std.sys import argv

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.deep_agents.sac import SAC
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.envs.robots.so_arm101_xml import SoArm101Model
from noeira.envs.robots.so_arm101 import SoArm101ReachConfig
from noeira.core.run import resolve_checkpoint


comptime EnvT = Phyics3dEnv[
    SoArm101Model, SoArm101ReachConfig, DT, TERMINATE_ON_UNHEALTHY=False
]
comptime OBS_DIM = EnvT.OBS_DIM  # 27 (incl. the previous action)
comptime ACT_DIM = EnvT.ACTION_DIM  #  6
comptime HIDDEN = 256
comptime BATCH = 256
# Not in the checkpoint and unused by greedy eval — keep it small so the CPU
# allocation stays trivial.
comptime REPLAY_CAPACITY = 100_000

comptime CHECKPOINT_PATH = "sac_so_arm101_reach.ckpt"
comptime ACTION_SCALE = Scalar[DT](1.0)  # normalized; MUST match the trainer

comptime NUM_EPISODES = 10
comptime MAX_STEPS = 500  # SoArmReachConfig.MAX_STEPS
comptime FRAME_DELAY_MS = 16  # ~60 FPS playback


def ckpt_from_argv(default: String) raises -> String:
    """`--ckpt <run_id|path>`, else `default`. A RUN ID resolves to its
    `checkpoints/last.ckpt` — what `sac_so_arm101_reach_training_gpu.mojo`
    writes into its run directory."""
    var args = argv()
    for i in range(1, len(args) - 1):
        if String(args[i]) == "--ckpt":
            return resolve_checkpoint(String(args[i + 1]), String("last"))
    return default


def main() raises:
    seed(42)
    var ckpt = ckpt_from_argv(String(CHECKPOINT_PATH))
    print("=" * 70)
    print("SAC — SO-ARM101 reach, CPU eval + 3D rendering")
    print("=" * 70)
    print("  OBS_DIM         =", OBS_DIM)
    print("  ACT_DIM         =", ACT_DIM, "(joint angles in RADIANS)")
    print("  action_scale    =", ACTION_SCALE)
    print("  Checkpoint      =", ckpt)
    print("  Episodes        =", NUM_EPISODES)
    print("=" * 70)

    var agent = SAC["cpu", OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY, HIDDEN](
        action_scale=ACTION_SCALE,
    )

    print("Loading checkpoint...")
    try:
        agent.load(ckpt)
        print("Checkpoint loaded.")
    except e:
        print("ERROR loading checkpoint:", e)
        print("Train first:")
        print(
            "  pixi run -e nvidia mojo run -I ."
            " examples/so101/sac_so_arm101_reach_training_gpu.mojo"
        )
        return
    print()

    print("-" * 70)
    var ctx = DeviceContext()
    var env = EnvT(ctx)

    # ⚠ OPEN ON THE FREE CAMERA. `so_arm101.xml` declares exactly one camera —
    # `<camera name="wrist_cam">`, bolted to the wrist — and the model renderer
    # starts at `active_camera = 0`, so this used to open looking down the
    # gripper: an onboard view that whips around with the wrist and cannot be
    # dragged back, since a body-attached camera is re-aimed EVERY frame
    # (`model_renderer.render`). Press `1` in the window for the wrist view.
    #
    # Done HERE because `eval_render` owns the loop and calls `init_renderer`
    # itself; `_init_renderer` returns early when already initialised, so this
    # is one window, not two, and the only place to get in front of it.
    #
    # For a camera plus the reward as a live sparkline, pause/step and
    # recording, use `sac_so_arm101_reach_policy_viewer.mojo` instead — same
    # checkpoint, same env, with the ImGui sidebar around it.
    if env.init_renderer():
        env.renderer_request_free_camera()

    var avg_reward = Float64(
        agent.eval_render[EnvT](
            env,
            NUM_EPISODES,
            max_steps_per_episode=MAX_STEPS,
            frame_delay_ms=FRAME_DELAY_MS,
            verbose=True,
        )
    )

    print("-" * 70)
    print("EVAL SUMMARY — SO-ARM101 reach")
    print("-" * 70)
    print("  Episodes        =", NUM_EPISODES)
    print("  Average reward  =", avg_reward)
    # Reward is a shaped `tolerance` in [0, 1] per step over MAX_STEPS control
    # steps, so the ceiling is 500 and what scores is reaching EARLY and
    # HOLDING.
    #
    # ⚠⚠ THE BANDS ARE ANCHORED ON A MEASURED BASELINE, NOT ON ROUND NUMBERS.
    # An UNTRAINED SAC actor — random init, greedy action — was run through
    # this exact env for 11 episodes (2026-08-26): mean **45.9**, per-episode
    # 3.5 .. 123.8. The margin is 0.25 m against a 0.15-0.30 m target shell, so
    # an arm that flails near the middle of its workspace collects real reward
    # for free. The earlier "> 50 = PROGRESS" band was therefore INDISTIN-
    # GUISHABLE FROM AN UNTRAINED NET, and a single episode proves nothing at
    # all: one untrained episode scored 123.8.
    # ⚠⚠ AND THE BAND NAMES ARE STILL A CLAIM THE RETURN CANNOT SUPPORT.
    # `tolerance`'s margin (0.25 m) is TWELVE TIMES its success radius
    # (0.02 m), so hovering 40 mm away for a whole episode scores 492.7 / 500
    # — above any episode this checkpoint has actually produced. A mean of 475
    # is therefore consistent with an arm that never touches the target, and
    # measurement says roughly half its episodes do not: mean CLOSEST approach
    # 20 mm, mean FINAL distance 38 mm.
    #
    # `examples/so101/sac_so_arm101_reach_diag.mojo` reports reached / held /
    # command chatter, which are the questions this number cannot answer. Read
    # the bands below as "how much reward", never as "did it reach".
    #
    # ⚠⚠ AND EVERY NUMBER IN THIS COMMENT IS NOW ON AN OBSOLETE SCALE. The
    # task gained a multiplicative stillness term (`VEL_FLOOR`, so a
    # permanently vibrating arm forfeits 20% of every step) and `R_MIN` moved
    # 0.15 -> 0.18. Measured: the SAME checkpoint that scored 480.5 under the
    # old reward scores **388.5** under the new one — the policy did not get
    # worse, the ruler got stricter. The 46 untrained baseline and the 90 /
    # 200 / 400 thresholds all predate both changes. RE-MEASURE THEM AFTER THE
    # NEXT RETRAIN; until then these bands are indicative only.
    if avg_reward > 400.0:
        print("  Result: high reward (mean > 400 / 500) — NOT the same as")
        print("          reaching; run ..._reach_diag.mojo for reach + hold.")
    elif avg_reward > 200.0:
        print("  Result: moderate reward (mean > 200) — see ..._reach_diag.")
    elif avg_reward > 90.0:
        print(
            "  Result: PROGRESS — measurably above the untrained baseline"
            " (~46)."
        )
    else:
        print(
            "  Result: NO BETTER THAN AN UNTRAINED NET (baseline mean ~46)."
        )
    print("=" * 70)
