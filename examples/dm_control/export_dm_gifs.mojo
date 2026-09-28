"""Export GIF recordings of DeepMind Control tasks with random actions.

The dm_control counterpart of `examples/export_env_gifs.mojo`: each task runs
for a fixed number of steps with uniform random actions and is recorded into
`gifs/dm_<domain>.gif`, which `docs-site/scripts/build-media.mjs` turns into
the documentation's demo clips.

    pixi run -e apple mojo run -I . examples/dm_control/export_dm_gifs.mojo            # all
    pixi run -e apple mojo run -I . examples/dm_control/export_dm_gifs.mojo walker dog # some

Random actions, not policies: the clips show the models, their contacts and
their cameras, not what a trained agent does with them. Each task is filmed
through one of its model's own cameras, chosen per task below.
"""

from std.random import random_float64, seed
from std.sys import argv

from noeira.envs.phyics3d_env import Phyics3dEnv, Phyics3dEnvConfig
from noeira.physics3d.model import ModelDefLike

from noeira.envs.dm_control.walker import DMWalkerWalk
from noeira.envs.dm_control.cheetah import DMCheetahRun
from noeira.envs.dm_control.humanoid import DMHumanoidWalk
from noeira.envs.dm_control.quadruped import DMQuadrupedWalk
from noeira.envs.dm_control.dog import DMDogWalk
from noeira.envs.dm_control.finger import DMFingerSpin
from noeira.envs.dm_control.fish import DMFishSwim
from noeira.envs.dm_control.manipulator import DMManipulatorBringBall


comptime NAMES = "walker cheetah humanoid quadruped dog finger fish manipulator"


def record_env[
    MODEL: ModelDefLike, CONFIG: Phyics3dEnvConfig, DTYPE: DType, TERM: Bool
](
    mut env: Phyics3dEnv[MODEL, CONFIG, DTYPE, TERM],
    name: String,
    camera: Int,
    steps: Int,
    skip: Int,
) raises:
    """`render_random`, plus a model camera chosen before the first frame.

    The camera request is read by the next render, so one frame is drawn
    before recording starts — otherwise the clip opens on the default view
    and cuts to the chosen one.
    """
    var path = "gifs/dm_" + name + ".gif"
    print("Recording", name, "camera", camera, "->", path)
    _ = env.init_renderer(show_velocity=False)
    env.renderer_request_camera(camera)
    _ = env.reset_obs_list()
    env.render_frame()
    env.start_recording(path, 30, skip)

    var lo = Float64(env.action_low())
    var hi = Float64(env.action_high())
    for _ in range(steps):
        if env.check_renderer_quit() or not env.is_renderer_open():
            break
        var action = List[Float64](capacity=env.action_dim())
        for _ in range(env.action_dim()):
            action.append(random_float64(lo, hi))
        var result = env.step_continuous_vec(action)
        env.render_frame()
        if result[2]:
            _ = env.reset_obs_list()

    env.stop_recording()
    env.close_renderer()


def wanted(name: String) raises -> Bool:
    var args = argv()
    if len(args) <= 1:
        return True
    for i in range(1, len(args)):
        if String(args[i]) == name:
            return True
    return False


def main() raises:
    for i in range(1, len(argv())):
        var a = String(argv()[i])
        if (" " + a + " ") not in (" " + String(NAMES) + " "):
            raise Error("unknown task '" + a + "'")

    seed(1)
    if wanted("walker"):
        var env = DMWalkerWalk()
        record_env(env, "walker", 0, 500, 3)
    if wanted("cheetah"):
        var env = DMCheetahRun()
        record_env(env, "cheetah", 0, 500, 3)
    if wanted("humanoid"):
        var env = DMHumanoidWalk()
        record_env(env, "humanoid", 1, 500, 3)
    if wanted("quadruped"):
        var env = DMQuadrupedWalk()
        record_env(env, "quadruped", 2, 500, 3)
    if wanted("dog"):
        var env = DMDogWalk()
        record_env(env, "dog", 0, 500, 3)
    if wanted("finger"):
        var env = DMFingerSpin()
        record_env(env, "finger", 0, 400, 2)
    if wanted("fish"):
        var env = DMFishSwim()
        record_env(env, "fish", 2, 500, 3)
    if wanted("manipulator"):
        var env = DMManipulatorBringBall()
        record_env(env, "manipulator", 0, 400, 2)
