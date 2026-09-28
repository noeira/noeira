"""Render a PPO LunarLander (discrete) checkpoint to a GIF or an MP4.

Loads a checkpoint written by `ppo_lunar_lander_time_to_solve.mojo` (CPU,
Metal or CUDA — the file is target-agnostic) into a CPU agent and records
greedy episodes. The format follows the output extension (ffmpeg). The
defaults record the way the old `gifs/lunar_lander_ppo_trained.gif` was
(30 fps, every 3rd frame, 3 episodes); for real time pass fps 50, skip 1.

Args (positional): checkpoint_path [out_path] [episodes=3] [env_seed=7]
                   [fps=30] [skip=3]

    pixi run mojo run -I . examples/lunar_lander/ppo_lunar_lander_render_gif.mojo \
        ppo_ll_cuda_s1.ckpt gifs/lunar_lander_ppo_discrete_trained.gif
    pixi run mojo run -I . examples/lunar_lander/ppo_lunar_lander_render_gif.mojo \
        ppo_ll_cuda_s1.ckpt gifs/lunar_lander_ppo_discrete_trained.mp4 3 7 50 1
"""

from std.random import seed
from std.sys import argv

from noeira.nn.constants import DT
from noeira.deep_agents.ppo_discrete import PPODiscreteAgent
from noeira.deep_agents.ppo_discrete.config import (
    PPODiscreteActorNet,
    PPODiscreteCriticNet,
)
from noeira.envs.lunar_lander import LunarLander


# Must match the trained agent's nets (ppo_lunar_lander_time_to_solve.mojo).
comptime OBS_DIM = 8
comptime N_ACTIONS = 4
comptime HIDDEN = 64


def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error(
            "usage: ppo_lunar_lander_render_gif.mojo checkpoint [out]"
            " [episodes] [seed] [fps] [skip]"
        )
    var ckpt_path = String(args[1])
    var gif_path = (
        String(args[2]) if len(args) > 2
        else String("gifs/lunar_lander_ppo_discrete_trained.gif")
    )
    var episodes = Int(String(args[3])) if len(args) > 3 else 3
    var env_seed = Int(String(args[4])) if len(args) > 4 else 7
    var fps = Int(String(args[5])) if len(args) > 5 else 30
    var skip = Int(String(args[6])) if len(args) > 6 else 3
    seed(env_seed)

    var agent = PPODiscreteAgent[
        "cpu",
        PPODiscreteActorNet[OBS_DIM, N_ACTIONS, HIDDEN],
        PPODiscreteCriticNet[OBS_DIM, HIDDEN],
        OBS_DIM, N_ACTIONS, 1, 1, 1,
    ]()
    agent.load(ckpt_path)
    print("Loaded", ckpt_path)

    var env = LunarLander[DT](seed=UInt64(env_seed))
    _ = env.init_renderer()
    env.start_recording(gif_path, fps=fps, skip=skip)

    for ep in range(episodes):
        var obs = env.reset_obs_list()
        var total: Float64 = 0.0
        for _ in range(1000):
            var action = agent.select_greedy_action(obs)
            var res = env.step_obs(action)
            total += Float64(res[1])
            env.render_frame()
            if env.check_renderer_quit() or res[2]:
                break
            obs = res[0].copy()
        print("  episode", ep + 1, "| return", total)

    env.stop_recording()
    env.close_renderer()
    print("Saved:", gif_path)
