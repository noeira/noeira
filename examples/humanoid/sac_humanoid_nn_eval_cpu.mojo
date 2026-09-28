"""Evaluate + render a trained Humanoid SAC (deep_agents) checkpoint.

Loads the one-file `nn-ckpt v2` checkpoint written by the Humanoid SAC GPU
trainer (`sac_humanoid_training_gpu.mojo`) and runs deterministic (greedy /
actor-mean, no sampling) episodes with live 3D rendering via the physics3d
`RenderableEnv` interface — the same renderer the legacy
`sac_humanoid_eval_cpu.mojo` drives, but on the new `SACAgent` facade.

Both the trainer and this eval build the agent through the shared `SAC[...]`
preset (`deep_agents.sac`), so the actor/critic architecture — and therefore
the checkpoint's parameter layout — matches exactly (fused `LinearReLU` hidden
layers, HIDDEN=256). `REPLAY_CAPACITY` here is deliberately small: the replay
buffer is NOT part of the checkpoint, so the eval can use a tiny buffer even
though training used 1M.

IMPORTANT: Humanoid trains with `action_scale=0.4`. The greedy action is
scaled by `action_scale` (`a = tanh(μ) · action_scale`), so this eval MUST
construct the agent with the same `action_scale=0.4` to reproduce the trained
policy's torques — unlike Walker2d where the default scale of 1.0 applies.

Controls (renderer window):
  * close the window or press its quit key to stop early.

Run:
    pixi run mojo run -I . examples/humanoid/sac_humanoid_nn_eval_cpu.mojo
    pixi run mojo run -I . examples/humanoid/sac_humanoid_nn_eval_cpu.mojo --ckpt <run_id>
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

# Per-field tensor physics path (migration P5+): the fields facade renders
# via the same physics3d ModelRenderer, driven by the bridge FK poses.
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

# Evaluation settings
comptime NUM_EPISODES = 10
comptime MAX_STEPS = 1000
comptime FRAME_DELAY_MS = 16  # ~60 FPS playback


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
    print("SAC (deep_agents) — Humanoid CPU eval + 3D rendering")
    print("=" * 70)
    print("  OBS_DIM         =", OBS_DIM)
    print("  ACT_DIM         =", ACT_DIM)
    print("  HIDDEN          =", HIDDEN)
    print("  action_scale    =", ACTION_SCALE)
    print("  Checkpoint      =", ckpt)
    print("  Episodes        =", NUM_EPISODES)
    print("  Max steps/ep    =", MAX_STEPS)
    print("=" * 70)

    # ─── Agent (architecture comes from the SAC preset — matches training) ─
    # Same `SAC[...]` preset the trainer uses, so the loaded checkpoint's
    # parameter layout matches exactly. `action_scale=0.4` is required for the
    # greedy action to reproduce the trained policy (it scales the tanh output).
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

    # ─── Greedy eval + live 3D rendering ─────────────────────────────────
    # `eval_render` is the facade's render-enabled greedy eval: it owns the
    # init_renderer / per-step render+delay / quit / close handling (the
    # `RenderableEnv` loop that used to be inlined here) and returns the mean
    # episode return. Falls back to headless if no renderer is available.
    print("-" * 70)
    var ctx = DeviceContext()  # fields facade: host staging for the model bridge
    var env = EnvT(ctx)
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
    print("EVAL SUMMARY — Humanoid (SAC deep_agents)")
    print("-" * 70)
    print("  Episodes        =", NUM_EPISODES)
    print("  Average reward  =", avg_reward)
    if avg_reward > 5000.0:
        print("  Result: EXCELLENT — walking (mean > 5000).")
    elif avg_reward > 2000.0:
        print("  Result: STRONG — sustained upright locomotion (mean > 2000).")
    elif avg_reward > 1000.0:
        print("  Result: PROGRESS — staying upright (mean > 1000).")
    elif avg_reward > 0.0:
        print("  Result: LEARNING — positive return (mean > 0).")
    else:
        print("  Result: EARLY — still exploring (mean < 0).")
    print("=" * 70)
