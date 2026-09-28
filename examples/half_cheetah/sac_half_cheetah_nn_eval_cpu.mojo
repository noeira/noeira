"""Evaluate + render a trained HalfCheetah SAC (deep_agents) checkpoint.

Counterpart of `examples/humanoid/sac_humanoid_nn_eval_cpu.mojo` for
HalfCheetah. Loads the one-file `nn-ckpt v2` checkpoint written by a
HalfCheetah SAC trainer and runs deterministic (greedy / actor-mean, no
sampling) episodes with live 3D rendering through the physics3d
`RenderableEnv` interface.

The eval builds the agent through the shared `SAC[...]` preset
(`deep_agents.sac`), whose actor/critic architecture — fused `LinearReLU`
hidden layers, HIDDEN=256 — is IDENTICAL to the nets in
`sac_half_cheetah_training_gpu.mojo` / `sac_half_cheetah_training.mojo`, so
the checkpoint's parameter layout matches exactly. `REPLAY_CAPACITY` here is
deliberately small: the replay buffer is NOT part of the checkpoint, so the
eval can use a tiny buffer even though training used 100k–1M.

IMPORTANT: HalfCheetah trains with the DEFAULT `action_scale=1.0` (unlike
Humanoid, which uses 0.4). The greedy action is `a = tanh(μ) · action_scale`,
so this eval constructs the agent with `action_scale=1.0` to reproduce the
trained policy's torques.

Physics: the env is `Phyics3dEnv` (per-field tensor path, migration
P5+) with `TERMINATE_ON_UNHEALTHY=False` — HalfCheetah has no early
termination; episodes run the full `MAX_STEPS`. Rendering is driven off the
bridge FK poses the fields step re-syncs each frame.

Controls (renderer window):
  * close the window or press its quit key to stop early.

Run:
    pixi run mojo run -I . examples/half_cheetah/sac_half_cheetah_nn_eval_cpu.mojo
    pixi run mojo run -I . examples/half_cheetah/sac_half_cheetah_nn_eval_cpu.mojo --ckpt <run_id>
"""

from std.sys import argv
from noeira.core.run import resolve_checkpoint
from std.random import seed

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.deep_agents.sac import SAC
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.envs.half_cheetah import HalfCheetahModel, HalfCheetahConfig


# =============================================================================
# Architecture — comes from the `SAC[...]` preset (matches the trainer)
# =============================================================================

# Per-field tensor physics path (migration P5+): the fields facade renders
# via the same physics3d ModelRenderer, driven by the bridge FK poses.
comptime EnvT = Phyics3dEnv[
    HalfCheetahModel, HalfCheetahConfig, DT, TERMINATE_ON_UNHEALTHY=False
]
comptime OBS_DIM = EnvT.OBS_DIM  # 17
comptime ACT_DIM = EnvT.ACTION_DIM  #  6
comptime HIDDEN = 256
comptime BATCH = 256
# Replay capacity is irrelevant for greedy eval (no replay used) and is NOT in
# the checkpoint — keep it small to avoid a large CPU allocation.
comptime REPLAY_CAPACITY = 100_000

comptime CHECKPOINT_PATH = "sac_half_cheetah_nn.ckpt"
comptime ACTION_SCALE = Scalar[DT](1.0)  # HalfCheetah default (unlike Humanoid 0.4)

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
    print("SAC (deep_agents) — HalfCheetah CPU eval + 3D rendering")
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
    # Same `SAC[...]` preset architecture the trainer uses, so the loaded
    # checkpoint's parameter layout matches exactly. `action_scale=1.0` is the
    # HalfCheetah default and is required for the greedy action to reproduce
    # the trained policy (it scales the tanh output).
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
        print("Train first (GPU, writes a preset-compatible checkpoint):")
        print(
            "  pixi run -e nvidia mojo run -I ."
            " examples/half_cheetah/sac_half_cheetah_training_gpu.mojo"
        )
        return
    print()

    # ─── Greedy eval + live 3D rendering ─────────────────────────────────
    # `eval_render` is the facade's render-enabled greedy eval: it owns the
    # init_renderer / per-step render+delay / quit / close handling (the
    # `RenderableEnv` loop) and returns the mean episode return. Falls back to
    # headless if no renderer is available.
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
    print("EVAL SUMMARY — HalfCheetah (SAC deep_agents)")
    print("-" * 70)
    print("  Episodes        =", NUM_EPISODES)
    print("  Average reward  =", avg_reward)
    if avg_reward > 4000.0:
        print("  Result: EXCELLENT — running fast (mean > 4000).")
    elif avg_reward > 1000.0:
        print("  Result: STRONG — learned locomotion (mean > 1000).")
    elif avg_reward > 100.0:
        print("  Result: PROGRESS — early locomotion (mean > 100).")
    elif avg_reward > 0.0:
        print("  Result: LEARNING — positive return (mean > 0).")
    else:
        print("  Result: EARLY — still exploring (mean < 0).")
    print("=" * 70)
