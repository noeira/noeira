"""Record a GIF of a trained Humanoid SAC (deep_agents) checkpoint.

Loads the one-file `nn-ckpt v2` checkpoint written by the Humanoid SAC GPU
trainer (`sac_humanoid_training_gpu.mojo`) and records deterministic (greedy /
actor-mean, no sampling) episodes through the physics3d 3D renderer into a GIF.

Same agent/checkpoint identity as `sac_humanoid_nn_eval_cpu.mojo` — the
`SAC[...]` preset with HIDDEN=256 and `action_scale=0.4` (MUST match training:
the greedy action is `a = tanh(μ) · action_scale`).

Recording is armed on the env's renderer BEFORE `eval_render` runs: the
renderer captures every rendered frame, and `eval_render`'s final
`close_renderer` stops the recorder and flushes the GIF file.

Run:
    pixi run mojo run -I . examples/humanoid/sac_humanoid_nn_gif.mojo
    pixi run mojo run -I . examples/humanoid/sac_humanoid_nn_gif.mojo --ckpt <run_id>

Reads sac_humanoid_nn.ckpt.
Writes gifs/sac_humanoid.gif.
"""

from std.sys import argv
from noeira.core.run import resolve_checkpoint
from std.random import seed

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.deep_agents.sac import SAC
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.envs.humanoid.humanoid_xml import HumanoidModel
from noeira.envs.humanoid.humanoid_config import HumanoidConfig


# =============================================================================
# Architecture — comes from the `SAC[...]` preset (matches the trainer)
# =============================================================================

# Per-field tensor physics path (migration P5+): the fields facade renders +
# records via the same physics3d ModelRenderer, driven by the bridge FK poses.
comptime EnvT = Phyics3dEnv[
    HumanoidModel, HumanoidConfig, DT, TERMINATE_ON_UNHEALTHY=True
]
comptime OBS_DIM = EnvT.OBS_DIM  # 45
comptime ACT_DIM = EnvT.ACTION_DIM  # 17
comptime HIDDEN = 256
comptime BATCH = 256
# Replay capacity is irrelevant for greedy eval (no replay used) and is NOT in
# the checkpoint — keep it small to avoid a large CPU allocation.
comptime REPLAY_CAPACITY = 100_000

comptime CHECKPOINT_PATH = "sac_humanoid_nn.ckpt"
comptime ACTION_SCALE = Scalar[DT](0.4)  # MUST match training

comptime GIF_PATH = "gifs/sac_humanoid.gif"
comptime GIF_EPISODES = 1
comptime GIF_FPS = 30
comptime GIF_FRAME_SKIP = 2  # record every 2nd rendered frame
comptime MAX_STEPS = 1000
comptime FRAME_DELAY_MS = 0  # no playback pacing — record as fast as it steps


def _flag(name: String, dflt: String) raises -> String:
    """Value of `--name X`, or `dflt` when the flag is absent."""
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return dflt


def main() raises:
    var ckpt = resolve_checkpoint(_flag(String("--ckpt"), String(CHECKPOINT_PATH)), String("last"))
    seed(42)
    print("=" * 70)
    print("SAC (deep_agents) — Humanoid GIF export (3D renderer)")
    print("=" * 70)
    print("  OBS_DIM         =", OBS_DIM)
    print("  ACT_DIM         =", ACT_DIM)
    print("  action_scale    =", ACTION_SCALE)
    print("  Checkpoint      =", ckpt)
    print("  Episodes        =", GIF_EPISODES)
    print("  Output          =", GIF_PATH)
    print("=" * 70)

    # ─── Agent (architecture comes from the SAC preset — matches training) ─
    var agent = SAC["cpu", OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY, HIDDEN](
        action_scale=ACTION_SCALE,
    )

    # ─── Load checkpoint ─────────────────────────────────────────────────
    print("Loading checkpoint...")
    try:
        agent.load(ckpt)
        print("Checkpoint loaded.")
    except e:
        print("ERROR loading checkpoint:", e)
        print("Train first (GPU):")
        print(
            "  pixi run -e nvidia mojo run -I ."
            " examples/humanoid/sac_humanoid_training_gpu.mojo"
        )
        return
    print()

    # ─── Greedy eval + GIF recording ─────────────────────────────────────
    # Init the renderer HERE (eval_render's own init_renderer is idempotent)
    # so recording can be armed before the eval loop renders its first frame.
    # eval_render's final close_renderer stops the recorder and flushes the
    # GIF — no explicit stop_recording needed.
    print("-" * 70)
    var ctx = DeviceContext()  # fields facade: host staging for the model bridge
    var env = EnvT(ctx)
    if not env.init_renderer():
        print("ERROR: renderer unavailable — cannot record a GIF.")
        return
    env.start_recording(String(GIF_PATH), fps=GIF_FPS, skip=GIF_FRAME_SKIP)
    print("Recording", GIF_EPISODES, "episode(s) to", GIF_PATH, "...")

    var avg_reward = Float64(
        agent.eval_render[EnvT](
            env,
            GIF_EPISODES,
            max_steps_per_episode=MAX_STEPS,
            frame_delay_ms=FRAME_DELAY_MS,
            verbose=True,
        )
    )

    print("-" * 70)
    print("  Episodes        =", GIF_EPISODES)
    print("  Average reward  =", avg_reward)
    print("  Saved:", GIF_PATH)
    print("=" * 70)
