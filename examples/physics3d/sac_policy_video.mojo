"""Record an MP4 (or GIF) of a trained SAC policy on a physics3d env.

One script for every SAC checkpoint whose nets are the standard pair
(`StochasticActor` over two `LinearReLU` layers; twin `LinearReLU` critics,
i.e. `SACActorNet` / `SACCriticNet`): the Gym-style HalfCheetah and Humanoid,
and the dm_control walker and dog. The env is chosen at compile time, so only
one model is built:

    -D VIDEO_ENV=half_cheetah     Gym HalfCheetah       hidden 256, scale 1.0
    -D VIDEO_ENV=humanoid         Gym Humanoid          hidden 256, scale 0.4
    -D VIDEO_ENV=dm_walker_walk   dm_control walker     hidden 256, scale 1.0
    -D VIDEO_ENV=dm_walker_run    dm_control walker     hidden 256, scale 1.0
    -D VIDEO_ENV=dm_dog_walk      dm_control dog        hidden 1024, scale 1.0
    -D VIDEO_ENV=dm_dog_trot      dm_control dog        hidden 1024, scale 1.0
    -D VIDEO_ENV=dm_dog_run       dm_control dog        hidden 1024, scale 1.0
    -D VIDEO_HIDDEN=H             override the width (must match training)

The rollout is greedy (the actor's mean, no sampling) on the CPU env, rendered
through the physics3d renderer from one of the model's own cameras (all four
models have a `trackcom` camera at index 0), with the HUD and the velocity
arrow off. The frame rate defaults to real time, which is
`1 / (timestep * FRAME_SKIP * skip)`. `ffmpeg` picks the codec from the
extension.

Args (flags, all optional):
    --ckpt PATH      checkpoint (required in practice)
    --out PATH       output video (default gifs/<env>_sac.mp4)
    --episodes N     episodes back to back in one file (default 1)
    --steps N        max steps per episode (default 1000)
    --skip K         record every K-th step (default: 1, or 2 for humanoid)
    --fps F          override the real-time frame rate
    --camera C       model camera index, -1 = free camera (default 0)
    --seed S         reset seed (default 1)

    pixi run mojo build -I . -D VIDEO_ENV=half_cheetah \\
        examples/physics3d/sac_policy_video.mojo -o sac_video_hc
    ./sac_video_hc --ckpt checkpoints/sac_half_cheetah_2026-09-25/sac_hc_cpu_s1.ckpt
"""

from std.random import seed
from std.sys import argv
from std.sys.defines import get_defined_int, get_defined_string

from noeira.nn.constants import DT
from noeira.deep_agents.sac import SACAgent
from noeira.deep_agents.sac.config import SACActorNet, SACCriticNet
from noeira.deep_agents.training.blocks import UniformSampleCpuStep
from noeira.envs.phyics3d_env import Phyics3dEnv, Phyics3dEnvConfig
from noeira.physics3d.model import ModelDefLike
from noeira.envs.half_cheetah import HalfCheetahModel, HalfCheetahConfig
from noeira.envs.humanoid.humanoid_xml import HumanoidModel
from noeira.envs.humanoid.humanoid_config import HumanoidConfig
from noeira.envs.dm_control.walker import DMWalkerWalk, DMWalkerRun
from noeira.envs.dm_control.dog import DMDogWalk, DMDogTrot, DMDogRun


comptime VIDEO_ENV = get_defined_string["VIDEO_ENV", "half_cheetah"]()


def _flag(name: String, dflt: String) raises -> String:
    """Value of `--name X`, or `dflt` when the flag is absent."""
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return dflt


def record[
    MODEL: ModelDefLike,
    CONFIG: Phyics3dEnvConfig,
    DTYPE: DType,
    TERM: Bool,
    HIDDEN: Int,
](
    mut env: Phyics3dEnv[MODEL, CONFIG, DTYPE, TERM],
    action_scale: Scalar[DT],
    default_skip: Int,
) raises:
    comptime OBS = MODEL.OBS_DIM
    comptime ACT = MODEL.ACTION_DIM

    var ckpt = _flag("--ckpt", "")
    if ckpt.byte_length() == 0:
        raise Error("--ckpt PATH is required")
    var out = _flag("--out", "gifs/" + String(VIDEO_ENV) + "_sac.mp4")
    var episodes = Int(_flag("--episodes", "1"))
    var max_steps = Int(_flag("--steps", "1000"))
    var skip = Int(_flag("--skip", String(default_skip)))
    var control_dt = CONFIG.get_timestep() * Float64(CONFIG.FRAME_SKIP)
    var rt_fps = Int(1.0 / (control_dt * Float64(skip)) + 0.5)
    var fps = Int(_flag("--fps", String(rt_fps)))
    var camera = Int(_flag("--camera", "0"))
    seed(Int(_flag("--seed", "1")))

    print("=" * 70)
    print("SAC policy video —", VIDEO_ENV)
    print("  checkpoint   =", ckpt)
    print("  obs / act    =", OBS, "/", ACT, "| hidden", HIDDEN,
          "| action_scale", action_scale)
    print("  control dt   =", control_dt, "s | skip", skip, "| fps", fps,
          "(real time", rt_fps, ")")
    print("  camera       =", camera)
    print("  output       =", out)
    print("=" * 70)

    var agent = SACAgent[
        "cpu",
        UniformSampleCpuStep[OBS, ACT, 256, 1024],
        SACActorNet[OBS, ACT, HIDDEN],
        SACCriticNet[OBS, ACT, HIDDEN],
    ](action_scale=action_scale)
    agent.load(ckpt)

    if not env.init_renderer(show_velocity=False):
        raise Error("renderer unavailable")
    env.renderer_set_show_hud(False)
    if camera < 0:
        env.renderer_request_free_camera()
    else:
        env.renderer_request_camera(camera)
    # One frame before recording, so the clip opens on the chosen camera.
    _ = env.reset_obs_list()
    env.render_frame()
    env.start_recording(out, fps, skip)

    var obs = List[Scalar[DT]](capacity=OBS)
    var action = List[Scalar[DT]](capacity=ACT)
    for _ in range(ACT):
        action.append(0)
    var total: Float64 = 0.0
    for ep in range(episodes):
        var o = env.reset_obs_list()
        obs.clear()
        for i in range(OBS):
            obs.append(Scalar[DT](o[i]))
        var ret: Float64 = 0.0
        var steps = 0
        for _ in range(max_steps):
            if env.check_renderer_quit() or not env.is_renderer_open():
                break
            agent.select_greedy_action(obs, action)
            var r = env.step_continuous_vec(action)
            env.render_frame()
            ret += Float64(r[1])
            steps += 1
            for i in range(OBS):
                obs[i] = r[0][i]
            if r[2]:
                break
        total += ret
        print("  episode", ep + 1, ": return", ret, "over", steps, "steps")

    env.stop_recording()
    env.close_renderer()
    print("RESULT env=" + String(VIDEO_ENV),
          "ckpt=" + ckpt,
          "episodes=" + String(episodes),
          "mean_return=" + String(total / Float64(episodes)),
          "video=" + out)


def main() raises:
    comptime if VIDEO_ENV == "half_cheetah":
        var env = Phyics3dEnv[
            HalfCheetahModel, HalfCheetahConfig, DT, TERMINATE_ON_UNHEALTHY=False
        ]()
        record[HIDDEN=get_defined_int["VIDEO_HIDDEN", 256]()](
            env, Scalar[DT](1.0), 1
        )
    elif VIDEO_ENV == "humanoid":
        var env = Phyics3dEnv[
            HumanoidModel, HumanoidConfig, DT, TERMINATE_ON_UNHEALTHY=True
        ]()
        record[HIDDEN=get_defined_int["VIDEO_HIDDEN", 256]()](
            env, Scalar[DT](0.4), 2
        )
    elif VIDEO_ENV == "dm_walker_walk":
        var env = DMWalkerWalk[DT]()
        record[HIDDEN=get_defined_int["VIDEO_HIDDEN", 256]()](
            env, Scalar[DT](1.0), 1
        )
    elif VIDEO_ENV == "dm_walker_run":
        var env = DMWalkerRun[DT]()
        record[HIDDEN=get_defined_int["VIDEO_HIDDEN", 256]()](
            env, Scalar[DT](1.0), 1
        )
    elif VIDEO_ENV == "dm_dog_walk":
        var env = DMDogWalk[DT]()
        record[HIDDEN=get_defined_int["VIDEO_HIDDEN", 1024]()](
            env, Scalar[DT](1.0), 1
        )
    elif VIDEO_ENV == "dm_dog_trot":
        var env = DMDogTrot[DT]()
        record[HIDDEN=get_defined_int["VIDEO_HIDDEN", 1024]()](
            env, Scalar[DT](1.0), 1
        )
    elif VIDEO_ENV == "dm_dog_run":
        var env = DMDogRun[DT]()
        record[HIDDEN=get_defined_int["VIDEO_HIDDEN", 1024]()](
            env, Scalar[DT](1.0), 1
        )
    else:
        comptime assert False, (
            "VIDEO_ENV must be half_cheetah | humanoid | dm_walker_walk |"
            " dm_walker_run | dm_dog_walk | dm_dog_trot | dm_dog_run"
        )
