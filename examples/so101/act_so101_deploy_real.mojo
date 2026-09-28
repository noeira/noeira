# +--------------------------------------------------------------------------+ #
# | ACT on the PHYSICAL SO-101 — closed-loop, from two live cameras
# +--------------------------------------------------------------------------+ #
"""Drive the follower arm from the ACT policy, observing the world through the
same two cameras the demonstrations were recorded with.

    pixi run build-opencv                     # ONCE
    pixi run build-serial                     # ONCE

    # SAFE BY DEFAULT: reads the arm and the cameras, runs the policy, prints
    # every command it WOULD have sent, and never energises anything.
    pixi run mojo build -I . -Xlinker -ld_classic -o /tmp/act_deploy \\
        examples/so101/act_so101_deploy_real.mojo

    # a promoted policy: weights + policies/act.norm.json, pulled with
    #   pixi run project-pull so101-tower --weights
    /tmp/act_deploy --project so101-tower --role act --devices 0,1

    # a training run's own weights: a RUN ID (its checkpoints/best.ckpt, with
    # norm.json beside it) or a .ckpt path
    /tmp/act_deploy --ckpt <run_id> --devices 0,1

    # --arm is what actually moves the robot. Be at the desk, hand on the power.
    /tmp/act_deploy --project so101-tower --role act --devices 0,1 --arm --seconds 30

    # --store <the .h5 it was trained on> is OPTIONAL now: it enables check 1
    # (replaying a held-out episode) and --snap, which compares each camera
    # with the store's frames — worth it before the first armed run on a rig
    # whose cameras have been moved.
    /tmp/act_deploy --project so101-tower --store <...> --snap /tmp/snap

    # ON THE JETSON. Cameras by PATH (the udev symlinks — both cameras report
    # the same serial, so an index is not a name), the arm ports resolved from
    # the platform defaults, and the policy on the GPU:
    pixi run -e jetson act-deploy-jetson -- --project so101-tower \
        --devices /dev/soarm_cam_overhead,/dev/soarm_cam_wrist

⚠ THE INFERENCE DEVICE IS A BUILD-TIME CHOICE: `-DACT_GPU=1` (see
`DEPLOY_TARGET`). `--devices` takes camera indices OR device paths, `--port`
names the follower (else `$SOARM_FOLLOWER_PORT`, else the platform default),
and `--fourcc` requests a pixel format from a path-opened V4L2 camera.
`--undistort DIR` brings each fisheye frame to the SIMULATOR's pinhole camera
(`vision/fisheye.mojo`, `DIR/camera_<name>.txt` from
`examples/vision/calibrate_fisheye.mojo`) on the camera thread before the
resize — required for a policy trained on sim frames or on a store imported
with `--undistort`, and wrong for one trained on raw recordings.

⚠ The normalization comes from `norm.json` beside the weights
(`policies/<role>.norm.json`, or `<run>/checkpoints/norm.json` for `--ckpt`),
`--norm` names another, and `--store` recomputes it — both given must agree.

This is the closed-loop counterpart of `act_so101_openloop_eval.mojo` and the
ACT counterpart of `deploy_reach_real.mojo`. Where the reach deployment ran a
SIM-trained policy and had to synthesise its observation from a physics model,
this one runs a policy trained on REAL demonstrations and its observation is
the real thing: two camera frames and the follower's own measured pose.

⚠⚠ **`--arm` MOVES THE FOLLOWER, AND NOTHING ELSE DOES.** The opt-in is a scar:
on 2026-08-31 `record.mojo` armed the follower with nobody at the desk because
the dangerous behaviour was the default and the flag was the safe one. Read
`docs/SO101_SERIAL_LAYER.md` §safety before the first armed run.

⚠⚠ **A `finally` DOES NOT COVER AN ABORT OR A SIGNAL.** If this dies hard the
follower is left holding its pose — the recovery is `pixi run soarm-torque-off`
and the power switch, not the `finally`.

## ⚠⚠ The measured fact this program is built around: THE QUERY RATE

BATCH=1, the checkpoint's own dims, warm.

⚠ THE ROWS ARE NOT THE SAME EXPERIMENT, and saying so is the point. Only the
Orin row is a full 30 s CONTROL LOOP with two cameras and an arm attached; the
M1 rows are what a build prints during its start-up warm-up, on a machine that
never ran this loop. Do not subtract across them.

    Orin NX, CUDA  — closed loop, per query, mean:
        cameras 2.2 + preprocess 3.4 + forward 27.6 = 33.2 ms -> 30.0 Hz

    M1 Pro         — forward only, 5-query warm-up:
        CPU  55.5 ms        Metal 302.7 ms

    M1 Pro         — as measured when this file was first written, in a loop:
        CPU  95 ms -> ~10.5 Hz    Metal 155 ms -> ~6.5 Hz

⚠ THE TWO M1 SETS DISAGREE (55.5 vs 95, 302.7 vs 155) and neither is discarded:
they were taken differently — a warm-up burst right after `load` against a
sustained control loop — and the second is the one with cameras and an arm
competing for the machine. A number is only comparable to one taken the same
way.

**Metal is SLOWER THAN ITS OWN CPU, which is why a Mac build stays CPU-only.**
At BATCH=1 the graph is a few hundred tiny kernels and Metal pays a
command-buffer retirement per launch (~20 us floor,
`_the_metal_launch_floor_is_command_buffer_retirement`); there is not enough
work per kernel to hide it. ⚠ THAT SAYS NOTHING ABOUT CUDA: the same argument
ran the other way on the Orin, where the forward is 27.6 ms against the ARM
CPU's much worse. The device is a build-time choice (`-DACT_GPU=1`,
`DEPLOY_TARGET`) and each binary prints its own number before anything is
armed. Measure, do not assume.

⚠⚠ **30.0 Hz ON THE ORIN MEANS THE POLICY IS QUERIED AT THE CADENCE IT WAS
TRAINED AT — a waypoint every 1.0 demonstrated steps.** Getting there took two
changes that were each necessary and neither sufficient: the ACT preprocess
moved onto the camera thread (work per query 36.4 -> 31.0 ms, under the 33.3 ms
frame period), and `exposure_dynamic_framerate=0` on the overhead camera, which
was silently delivering 20 fps and PACING THE WHOLE LOOP — the resize saving
was absorbed to 0.1 ms by a longer camera wait until that was fixed. The
achieved 33.2 ms against a 33.3 ms period is a SATURATED loop: the camera is
now the limit, as it should be.

The policy was trained on 30 fps demonstrations, so a chunk's entries are
1/30 s apart. **On a machine that cannot query every step this program does not
pretend to** — the numbers above show which machines those are. Instead:

  * there is a 30 Hz ACTION GRID, defined by wall clock — `t = round(elapsed *
    30)` — and the chunk index means exactly what it meant in training;
  * an observation captured at grid step `t_obs` is queried, and the chunk it
    returns is pushed into the temporal ensemble AS `t_obs`;
  * by the time the forward returns, the clock has moved on ~3 steps, so what
    is COMMANDED is the ensemble's action for `t_cmd` — the grid step it is
    actually now. The inference latency is not hidden, it is INDEXED.

⚠ THE CONSEQUENCE, STATED PLAINLY: on a slow machine the arm receives a
waypoint every ~3 demonstrated steps, not every step. The trajectory plays at
the right SPEED (the grid is wall clock, so the policy does not run in slow
motion), but between waypoints the servo interpolates instead of the policy.
Temporal ensembling is what makes that tolerable — every commanded action is a
weighted blend of overlapping chunks — and the `ensemble` figure in the report
is how you check it is filling up: 38.7 of K=60 at 20 Hz, **58.0 of 60 at 30
Hz**, where the window is nearly saturated and the blend is at its smoothest.
At 30 Hz the interpolation gap closes entirely.

⚠ ONE SEMANTIC CONSEQUENCE OF QUERYING SPARSELY, recorded in
`TemporalEnsemble.action_at`: the ensemble weight `exp(-m*(i - i_min))` is the
paper's rank-based `w_i` only when every step in the window was queried. Here
it becomes a weight on AGE IN GRID STEPS instead — the right generalisation,
and with `m = 0.01` over `K = 60` a few percent either way.

Getting to a true 30 Hz needs the forward OFF the control thread, not a faster
device. That is a real design (the camera and encoder threads already prove the
machinery) and it is deliberately not attempted here: a first closed-loop
bring-up should not also be the debut of a cross-thread inference pipeline.

## What is checked BEFORE anything is armed

1. **the checkpoint and the store agree.** Normalization statistics are part of
   the policy and the checkpoint does not carry them — they are recomputed from
   the store, exactly as training did. Deploying with the WRONG store is
   therefore silent and dangerous: every command comes out shifted and scaled.
   So `--check` replays a held-out episode through the policy and reports the
   error against the same `hold` and `mean` baselines the open-loop evaluator
   uses. Worse than the constant `mean` baseline refuses to arm (`--force`
   overrides).
2. **the arm is somewhere the demonstrations went.** The dataset's own qpos box
   is printed against the follower's present pose, per joint. A joint parked
   outside it means the policy's first command is an extrapolation.
3. **the cameras are the right way round.** Slot 0 and slot 1 are NOT
   interchangeable — the store's camera order is alphabetical by feature key
   (`observation.images.front` then `...side`), and swapping them feeds the
   policy a world it has never seen. The mapping is printed; check it.

## Safety, on top of the two `SO101Arm` already enforces

`SO101Arm.write_goals` clamps to the calibrated `[range_min, range_max]` and to
`present ± max_step_ticks`. This program adds:

  4. the goal is parked on the follower's OWN present pose before torque is
     armed, so arming holds instead of snapping to a stale `Goal_Position`;
  5. a partial `sync_read` skips the tick's WRITE rather than commanding a
     half-updated pose;
  6. **the commanded action is clamped to the dataset's own action box** —
     the per-joint min and max the demonstrations ever reached. The policy has
     never been asked what to do outside it and its answer there is not
     evidence of anything. The report counts how often this fires; a clamp
     that fires constantly is telling you the deployment is off-distribution,
     not that the clamp is doing its job.
  7. **the run does not end by cutting torque where the policy left the arm.**
     The first armed run ended mid-reach and the arm DROPPED under its own
     weight. Releasing torque hands the arm to gravity, so where it is at that
     instant is a safety decision, not a detail. The shutdown now ramps back to
     the pose the run STARTED from — the one pose known to hold unpowered,
     because the arm was already resting there — confirms it arrived, holds,
     and releases on the operator's Enter. **If the ramp does not arrive,
     torque is LEFT ON**; a still-energised arm is recoverable with
     `pixi run soarm-torque-off`, and a fall is not. `--no-return` restores the
     old drop-where-it-stands behaviour and is for a run you are standing over.
"""

from std.os import getenv
from std.os.path import exists
from std.sys import argv
from std.sys.defines import is_defined
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.deep_agents.act.config import (
    ACT_TEMPORAL_ENSEMBLE_M,
    RUN_DEC_LAYERS,
    RUN_DIM,
    RUN_ENC_LAYERS,
    RUN_FF,
    RUN_HEADS,
    RUN_K,
    RUN_LATENT,
    SO101_ADIM,
    SO101_FPS,
    SO101_IMG_H,
    SO101_IMG_W,
    SO101_N_CAM,
    SO101_QPOS,
)
from noeira.deep_agents.act.data import ACTDataset
from noeira.data.lerobot import CameraStream, EpisodeIndex, LeRobotInfo
from noeira.deep_agents.act.norm_file import ACTNorm, act_norm_from
from noeira.io.image import resize_bilinear_pil
from noeira.deep_agents.act.inference import (
    TemporalEnsemble,
    denormalize,
    normalize_camera_chw,
)
from noeira.deep_agents.act.trainer import ACTTrainer
from noeira.io.fileio import StdinReader, stdin_is_tty
from noeira.io.json import load_json
from noeira.io.png import save_png
from noeira.robot.so101 import SO101Arm, SO101_N, joint_name
from noeira.robot.so101.ports import follower_port, port_refusal
from noeira.robot.so101.deploy_shutdown import (
    RETURN_TIMEOUT_S, return_and_release,
)
from noeira.utils.fmt import col, fixed, pad_left, pad_right
from noeira.vision.camera_thread import CameraReader, parse_camera_specs
from noeira.core.policy import describe_policy, resolve_policy
from noeira.core.run import resolve_checkpoint


comptime DEPLOY_TARGET: StaticString = "gpu" if is_defined["ACT_GPU"]() else "cpu"
"""⚠⚠ THE INFERENCE DEVICE, CHOSEN AT BUILD TIME — `-DACT_GPU=1`.

A BUILD-time switch and not a `--device` flag, because `target` is a parameter
of `ACTTrainer`: a runtime flag means instantiating the whole ACT graph TWICE
in one binary, and the baseline build is already 145 s on an M1 Pro (measured).
One target per binary costs nothing; two would be paid on every rebuild, on the
slowest machine in the loop.

⚠ THE DEFAULT IS CPU BECAUSE OF A MEASUREMENT, NOT A PREFERENCE — see the
header: on this M1, Metal at BATCH=1 was 155 ms against the CPU's 95 ms. That
number is about METAL's command-buffer retirement floor and it does NOT
transfer to CUDA. On the Jetson the comparison is a weak ARM CPU against a
1024-core Ampere, so GPU is the one to try — but `--seconds 0` prints the
forward latency of whichever binary you built, and THAT is the number to
believe. Build both, run both, read the two numbers."""

comptime POLICY_PROJECT = "so101"
comptime POLICY_ROLE = "act"
comptime DEFAULT_CKPT = "act_so101_best_gpu.ckpt"
"""⚠⚠ A FALLBACK, NOT THE DEPLOY TARGET — §8, decision 2. `--ckpt` still wins;
absent it, a checkpoint promoted into `projects/so101/policies/act.ckpt` is
used, and only if there is none does this flat constant apply.

⚠ Falling back rather than raising is deliberate: an arm that will not start
because the project layer is not set up yet is a regression for someone who
only wanted to run what worked yesterday."""

comptime QPOS = SO101_QPOS
comptime ADIM = SO101_ADIM
comptime N_CAM = SO101_N_CAM
comptime IMG_H = SO101_IMG_H
comptime IMG_W = SO101_IMG_W
# ⚠ FROM `act.config`, never restated. These ARE the checkpoint's parameter
# shapes; the open-loop evaluator once carried its own copy and drifted from
# the trainer, so a checkpoint the trainer told you to evaluate could not be
# loaded by the evaluator.
comptime K = RUN_K
comptime DIM = RUN_DIM
comptime HEADS = RUN_HEADS
comptime FF = RUN_FF
comptime LATENT = RUN_LATENT
comptime N_ENC = RUN_ENC_LAYERS
comptime N_DEC = RUN_DEC_LAYERS
comptime BATCH = 1

comptime T = ACTTrainer[
    QPOS, ADIM, N_CAM, IMG_H, IMG_W, K, DIM, HEADS, FF, LATENT, N_ENC,
    N_DEC, BATCH, target=DEPLOY_TARGET,
]
comptime CAM_ELEMS = 3 * IMG_H * IMG_W
comptime IMG_ELEMS = N_CAM * CAM_ELEMS

comptime UNDISTORT_FOVY = 73.7398
"""The sim cameras' fovy (`so101_tower_stand.xml` / `so_arm101_tower.xml`) —
the pinhole `--undistort` brings the fisheye to, as the importer does."""

comptime CAM_W = 640
comptime CAM_H = 480
"""⚠⚠ THE CAMERA'S NATIVE SIZE, AND IT MUST BE THE ONE THE DEMONSTRATIONS WERE
RECORDED AT — 480x640 for `DenisLabs/record-test_*` (`meta/info.json`
`observation.images.front.shape`). It is not a preference: PIL's bilinear
window GROWS with the reduction factor, so 640x480 -> 320x240 and 1280x720 ->
320x240 are different filters and produce different pixels from the same
scene. Feeding the second to a model trained on the first is a silent
train/deploy gap. `--width` / `--height` exist for a different rig, not for
convenience."""

comptime WARMUP_QUERIES = 5
"""Forwards run before anything is checked or armed — enough to separate the
first (which compiles kernels) from the steady state, few enough that a CPU
build at ~95 ms still starts in under a second."""

comptime MAX_STEP_TICKS = 80
"""~7 degrees per tick of the CONTROL loop. Same value `teleop.mojo` and
`record.mojo` measured with, and it is a bound on how far ahead of the arm a
goal may sit rather than a speed limit.

⚠ READ THE `throttle` LINE IN THE REPORT BEFORE BLAMING THE POLICY. This clamp
has NO COUNTERPART IN TRAINING — the demonstrations were recorded through it,
but the policy's output is not — so a clamp that binds is a deploy/train gap,
exactly as `deploy_reach_real.mojo` found the hard way (its first live run was
rate-limited by the clamp for its whole duration and looked like a bad policy)."""

comptime TRACK_STEP_TICKS = 512
"""~45 degrees: the clamp once the follower has caught up with the policy's
first goal. The SAME two phases the recorders use (`SO101Arm.track_step_ticks`),
so the arm the policy drives moves like the arm that recorded its
demonstrations: those were no longer lagging the leader by 300 ms behind an
80-tick clamp."""

comptime ACTION_BOX_MARGIN = 0.05
"""Widen the dataset's action box by 5% of its own span before clamping.

The box is a min/max over ~15,000 demonstrated frames, so its edges are real
poses the arm reached — clamping exactly at them would fight the policy at the
extremes of a legitimate motion (a fully closed gripper is AT the minimum). A
small margin keeps the guard about EXTRAPOLATION, which is what it is for."""


# ⚠ `return_and_release`, `RETURN_STEP_TICKS` and friends MOVED to
# `noeira/robot/so101/deploy_shutdown.mojo` when the SmolVLA deployment
# needed the same shutdown. Safety code that exists twice drifts; there is now
# one copy and both deployments call it.


def _split(s: String, sep: String) -> List[String]:
    var out = List[String]()
    var cur = String("")
    for i in range(s.byte_length()):
        var c = chr(Int(s.as_bytes()[i]))
        if c == sep:
            out.append(cur^)
            cur = String("")
        else:
            cur += c
    out.append(cur^)
    return out^


def camera_names(store: String) raises -> List[String]:
    """The store's camera keys, IN SLOT ORDER, from the sidecar if there is one.

    ⚠ SLOT ORDER IS ALPHABETICAL BY FEATURE KEY — that is how the importer
    assigns them (`data/lerobot.mojo`, `cameras` sorted by name), and it is
    the only thing that ties a physical camera to a channel of the tensor.
    The names are cosmetic; printing them is not, because "slot 0" tells an
    operator nothing and "slot 0 = observation.images.front" tells them
    which camera to check.
    """
    var out = List[String]()
    if not store.endswith(".h5"):
        return out^
    var side = store[byte=0 : store.byte_length() - 3] + ".json"
    if not exists(side):
        return out^
    try:
        var doc = load_json(side)
        var cams = doc.field(doc.root(), String("cameras"))
        if cams < 0:
            return out^
        for i in range(doc.size(cams)):
            out.append(doc.string(doc.at(cams, i)))
    except:
        # A sidecar we cannot read is not a reason to refuse to deploy; it is
        # a reason to print slot numbers instead of names.
        out = List[String]()
    return out^


def recorded_first_row[
    n_cam: Int, img_h: Int, img_w: Int
](root: String, ref cameras: List[String]) raises -> List[Scalar[DType.uint8]]:
    """The first frame of the recording's first episode, per camera, CHW —
    the same layout and the same resize the store holds.

    ⚠ THE SAME `resize_bilinear_pil` THE IMPORTER USED. A comparison through a
    different filter would show differences that are not there; this one shows
    only what the CAMERA changed.
    """
    var info = LeRobotInfo(root)
    var index = EpisodeIndex(root, info.cameras)
    var out = List[Scalar[DType.uint8]](length=n_cam * 3 * img_h * img_w, fill=0)
    var hwc = List[UInt8](unsafe_uninit_length = img_h * img_w * 3)
    var scratch = List[UInt8]()
    for c in range(n_cam):
        # ⚠ BY NAME, NOT BY POSITION: the store's slot order is the importer's
        # (alphabetical), and `cameras` carries it.
        var which = -1
        for k in range(len(info.cameras)):
            if info.cameras[k] == cameras[c]:
                which = k
        if which < 0:
            raise Error(
                "act deploy: the recording at " + root + " has no camera '"
                + cameras[c] + "'"
            )
        var stream = CameraStream(String(info.cameras[which]), String(root))
        var first = Int(round(index.vid_from_ts[which][0] * Float64(info.fps)))
        stream.open_at(index.vid_chunk[which][0], index.vid_file[which][0], first)
        stream.next_native()
        resize_bilinear_pil(
            stream.raw.unsafe_ptr().as_unsafe_any_origin(),
            stream.height,
            stream.width,
            hwc.unsafe_ptr().as_unsafe_any_origin(),
            img_h,
            img_w,
            scratch,
            3,
        )
        var base = c * 3 * img_h * img_w
        for ch in range(3):
            for i in range(img_h * img_w):
                out[base + ch * img_h * img_w + i] = Scalar[DType.uint8](
                    hwc[i * 3 + ch]
                )
        stream.close()
    return out^


def norm_beside(ckpt: String) -> String:
    """The norm.json that goes with a checkpoint, or "" if none exists.

    `policies/<role>.ckpt` -> `policies/<role>.norm.json` (what promotion
    writes); `<run>/checkpoints/<name>.ckpt` -> `<run>/checkpoints/norm.json`
    (what the trainer writes).
    """
    if ckpt.endswith(".ckpt"):
        var role_norm = String(ckpt[byte=0 : ckpt.byte_length() - 5]) + ".norm.json"
        if exists(role_norm):
            return role_norm
    var cut = ckpt.rfind("/")
    var run_norm = (String(ckpt[byte=0:cut]) if cut > 0 else String(".")) + "/norm.json"
    if exists(run_norm):
        return run_norm
    return String("")


def norm_candidates(ckpt: String) -> String:
    var cut = ckpt.rfind("/")
    var dir = String(ckpt[byte=0:cut]) if cut > 0 else String(".")
    if ckpt.endswith(".ckpt"):
        return String(ckpt[byte=0 : ckpt.byte_length() - 5]) + ".norm.json or " + dir + "/norm.json"
    return dir + "/norm.json"


def store_path() raises -> String:
    """`--store` or `$ACT_STORE` — there is deliberately no default.

    ⚠⚠ THE STORE IS PART OF THE POLICY. The normalization statistics are
    recomputed from it, and the checkpoint carries none — so pointing this at
    the wrong recording does not fail, it produces commands that are shifted
    and scaled wrong, on a real arm. Two stores in this cache differ only by a
    date in their name and one has 5 episodes where the other has 50. A
    default here would be a coin flip with the arm as the stake.
    """
    var env = getenv("ACT_STORE")
    if env.byte_length() > 0:
        return env^
    raise Error(
        "act deploy: name the store the checkpoint was TRAINED on, with"
        " --store <path.h5> or ACT_STORE. There is no default — see"
        " `store_path`'s note."
    )


def main() raises:
    var arm_it = False
    var force = False
    var store = String("")
    var ckpt = String("")
    var norm_file = String("")
    var project = String(POLICY_PROJECT)
    var role = String(POLICY_ROLE)
    var seconds = 30
    var step_ticks = MAX_STEP_TICKS
    var smooth = 1.0
    var check_steps = 30
    var devices = List[String]()
    var undistort_dir = String("")
    var port_arg = String("")
    var cam_fourcc = String("")
    var cam_w = CAM_W
    var cam_h = CAM_H
    var snap = String("")
    # A RECORDING directory for --snap, instead of the training store.
    #
    # ⚠ THE STORE IS 9 GB AND THE RECORDING IS ALREADY ON THE ROBOT MACHINE.
    # `projects/<p>/datasets/<d>` holds the same frames the store was built
    # from, so the camera comparison — the check that catches a swapped or
    # moved camera, which nothing else can — does not need the store copied
    # back from the training box.
    #
    # ⚠ A COMMENT, NOT A DOCSTRING: Mojo has no docstrings on LOCALS, so the
    # string was an unused expression and warned on every single build.
    var snap_from = String("")
    var do_return = True

    var args = argv()
    for i in range(len(args)):
        var a = String(args[i])
        if a == "--arm" or a == "--live":
            # `--live` is `deploy_reach_real.mojo`'s spelling and `--arm` is
            # `record.mojo`'s. Both mean "energise the follower"; accepting
            # both is cheaper than an operator discovering the difference by
            # typing the other one and getting a dry run they did not want.
            arm_it = True
        elif a == "--force":
            force = True
        elif a == "--store" and i + 1 < len(args):
            store = String(args[i + 1])
        elif a == "--ckpt" and i + 1 < len(args):
            ckpt = String(args[i + 1])
        elif a == "--norm" and i + 1 < len(args):
            norm_file = String(args[i + 1])
        elif a == "--project" and i + 1 < len(args):
            project = String(args[i + 1])
        elif a == "--role" and i + 1 < len(args):
            role = String(args[i + 1])
        elif a == "--seconds" and i + 1 < len(args):
            seconds = Int(String(args[i + 1]))
        elif a == "--step" and i + 1 < len(args):
            step_ticks = Int(String(args[i + 1]))
        elif a == "--smooth" and i + 1 < len(args):
            smooth = Float64(String(args[i + 1]))
        elif a == "--check" and i + 1 < len(args):
            check_steps = Int(String(args[i + 1]))
        elif a == "--width" and i + 1 < len(args):
            cam_w = Int(String(args[i + 1]))
        elif a == "--height" and i + 1 < len(args):
            cam_h = Int(String(args[i + 1]))
        elif a == "--no-return":
            # ⚠ THE ARM IS THEN RELEASED WHEREVER THE POLICY LEFT IT, which is
            # exactly how it fell the first time. For a run you are standing
            # over with a hand under the arm, nothing else.
            do_return = False
        elif a == "--snap" and i + 1 < len(args):
            snap = String(args[i + 1])
        elif a == "--snap-from" and i + 1 < len(args):
            snap_from = String(args[i + 1])
        elif a == "--devices" and i + 1 < len(args):
            # Indices, device PATHS, or a mix — see `parse_camera_specs`.
            devices = parse_camera_specs(String(args[i + 1]))
        elif a == "--undistort" and i + 1 < len(args):
            # A policy trained on a store imported with `--undistort DIR` (or
            # on the SIMULATOR's frames) sees the sim's pinhole camera, not
            # the raw fisheye: each frame goes through DIR/camera_<name>.txt
            # (`vision/fisheye.mojo`) on the camera thread, before the resize.
            undistort_dir = String(args[i + 1])
        elif a == "--port" and i + 1 < len(args):
            port_arg = String(args[i + 1])
        elif a == "--fourcc" and i + 1 < len(args):
            # ⚠ ONLY MEANINGFUL FOR A PATH-OPENED CAMERA (V4L2). Defaults to
            # MJPG there — `none` leaves the device alone. It must match what
            # the DEMONSTRATIONS were recorded in; see §3.2 of the Jetson
            # document for why a format change is a train/deploy gap.
            cam_fourcc = String(args[i + 1])
    if store == "":
        store = getenv("ACT_STORE")
    if undistort_dir.byte_length() > 0 and len(devices) == 0:
        raise Error("act deploy: --undistort needs --devices")
    if ckpt != "":
        # `--ckpt` takes a RUN ID too: the trainer writes
        # `runs/<id>/checkpoints/best.ckpt` with `norm.json` beside it, which
        # `norm_beside` below finds. A file path is used as given.
        ckpt = resolve_checkpoint(ckpt, String("best"))
    if ckpt == "":
        # ⚠ THE ROLE BEFORE THE CONSTANT, and an explicit `--ckpt` before both.
        ckpt = resolve_policy(project, role, String(DEFAULT_CKPT))
        var provenance = describe_policy(project, role)
        # ⚠ SAY WHICH WEIGHTS, BEFORE THE ARM MOVES. Silently loading something
        # other than what the operator expects is the failure this prevents.
        if provenance:
            print("  policy          = " + ckpt)
            print("  promoted from   = " + provenance)
        else:
            print(
                "  policy          = " + ckpt
                + "  (flat fallback; nothing promoted into the '"
                + role + "' role of project '" + project + "')"
            )
    if len(devices) == 0:
        devices.append(String("0"))
        devices.append(String("1"))
    if len(devices) != N_CAM:
        raise Error(
            "act deploy: the policy takes " + String(N_CAM) + " cameras but "
            + String(len(devices)) + " device(s) were given"
        )

    print("=" * 74)
    if arm_it:
        print("ACT / SO-101 — CLOSED LOOP ON THE REAL ARM   [ARMED]")
    else:
        print("ACT / SO-101 — DRY RUN (no torque, no goals written)")
        print("  pass --arm to actually move the follower")
    print("=" * 74)

    # ⚠⚠ THE NORMALIZATION: `--norm`, else the one beside the weights, else the
    # store. A checkpoint carries none of its own, and the wrong statistics do
    # not fail — they shift and scale every command on a real arm. See
    # `deep_agents/act/norm_file.mojo`.
    if norm_file == "" and store == "":
        norm_file = norm_beside(ckpt)
    if norm_file == "" and store == "":
        raise Error(
            "act deploy: no normalization for " + ckpt + ". Expected "
            + norm_candidates(ckpt) + ", or pass --norm <norm.json> or --store"
            " <the .h5 it was trained on>. For a run trained before norm.json"
            " existed: examples/so101/act_so101_export_norm.mojo."
        )
    var have_store = store != ""
    if have_store and not exists(store):
        raise Error("act deploy: no store at " + store)
    if snap != "" and not have_store and snap_from == "":
        raise Error(
            "act deploy: --snap compares each camera with what was RECORDED,"
            " so it needs --store <the .h5> or --snap-from"
            " projects/<project>/datasets/<dataset>"
        )
    if snap_from != "" and not exists(snap_from + "/meta/info.json"):
        raise Error("act deploy: no LeRobot recording at " + snap_from)
    if not exists(ckpt):
        raise Error(
            "act deploy: no checkpoint at " + ckpt + " — train first"
            " (examples/so101/act_so101_train_gpu.mojo) or pass --ckpt"
        )

    # ── the policy's units ────────────────────────────────────────────────
    # ⚠ `max_image_bytes=0` FORCES THE STREAMED PATH. The statistics come from
    # the qpos and action columns, which are a few hundred KB; the image
    # column of the 50-episode store is 7.1 GiB and residency would load all
    # of it to compute nothing. `--check` then streams the handful of rows it
    # actually reads.
    var ds_opt: Optional[ACTDataset[QPOS, ADIM, N_CAM, IMG_H, IMG_W]] = None
    var norm: ACTNorm
    if have_store:
        print("store       " + store)
        ds_opt = ACTDataset[QPOS, ADIM, N_CAM, IMG_H, IMG_W](
            store.copy(), seed=7, max_image_bytes=0
        )
        ref ds = ds_opt.value()
        print(
            "            " + String(ds.n_rows()) + " frames, "
            + String(ds.n_episodes()) + " episodes, "
            + String(len(ds.train_eps)) + " train / "
            + String(len(ds.val_eps)) + " held out"
        )
        var store_norm = act_norm_from(
            ds.qpos_raw, ds.action_raw, ds.n_rows(), ds.n_episodes(),
            ds.qpos_mean, ds.qpos_std, ds.action_mean, ds.action_std,
            camera_names(store), IMG_H, IMG_W, store,
        )
        if norm_file != "":
            # ⚠ BOTH GIVEN: they must agree, or one of them belongs to another
            # policy. The store is the ground truth for the check it enables.
            var file_norm = ACTNorm.load(norm_file, QPOS, ADIM)
            var same = True
            for j in range(QPOS):
                if file_norm.qpos_mean[j] != store_norm.qpos_mean[j] or file_norm.qpos_std[j] != store_norm.qpos_std[j]:
                    same = False
            for j in range(ADIM):
                if file_norm.action_mean[j] != store_norm.action_mean[j] or file_norm.action_std[j] != store_norm.action_std[j]:
                    same = False
            if not same and not force:
                raise Error(
                    "act deploy: " + norm_file + " does not match the statistics of "
                    + store + " — they come from different training data. Drop one,"
                    " or pass --force."
                )
        norm = store_norm^
    else:
        norm = ACTNorm.load(norm_file, QPOS, ADIM)
        print("normalization " + norm_file)
        print(
            "            " + String(norm.n_rows) + " frames, " + String(norm.n_episodes)
            + " episodes (from " + norm.store + ")"
        )

    # The action box: what the demonstrations ever commanded, per joint —
    # the same min/max over every row, now carried by `norm`.
    var a_lo = norm.action_min.copy()
    var a_hi = norm.action_max.copy()
    var q_lo = norm.qpos_min.copy()
    var q_hi = norm.qpos_max.copy()
    for j in range(ADIM):
        var pad = ACTION_BOX_MARGIN * (a_hi[j] - a_lo[j])
        a_lo[j] -= pad
        a_hi[j] += pad

    # ── the policy ────────────────────────────────────────────────────────
    print("checkpoint  " + ckpt)
    # ⚠ THE CONTEXT IS CREATED ONLY ON A GPU BUILD, and it must outlive `tr`:
    # the trainer's device buffers are allocated from it.
    var dev_ctx = Optional[DeviceContext](None)
    comptime if DEPLOY_TARGET != "cpu":
        dev_ctx = DeviceContext()
        print("device      " + String(dev_ctx.value().name()) + "  (-DACT_GPU=1)")
    else:
        print("device      CPU")
    var tr = T.make(ctx=dev_ctx)
    tr.load(ckpt)
    print(
        "            K=" + String(K) + " dim=" + String(DIM) + " enc="
        + String(N_ENC) + " dec=" + String(N_DEC) + "  ensemble m="
        + String(ACT_TEMPORAL_ENSEMBLE_M)
    )

    var qpos_n = List[Scalar[DT]](length=BATCH * QPOS, fill=Scalar[DT](0.0))
    var images_n = List[Scalar[DT]](
        length=BATCH * IMG_ELEMS, fill=Scalar[DT](0.0)
    )
    var dummy_actions = List[Scalar[DT]](
        length=BATCH * K * ADIM, fill=Scalar[DT](0.0)
    )
    var dummy_valid = List[Scalar[DT]](
        length=BATCH * K, fill=Scalar[DT](1.0)
    )
    var chunk = List[Scalar[DT]](length=BATCH * K * ADIM, fill=Scalar[DT](0.0))
    var pred_n = List[Scalar[DT]](length=ADIM, fill=Scalar[DT](0.0))
    var pred = List[Scalar[DT]](length=ADIM, fill=Scalar[DT](0.0))

    # ── warm-up: pay the first-forward cost HERE, not in the control loop ──
    #
    # ⚠⚠ THE FIRST FORWARD IS NOT THE PRICE OF THE OTHERS. On a GPU build it
    # compiles and caches every kernel it touches the first time it runs them —
    # measured at ~2.4 s for a small MLP on the Orin
    # (`docs/JETSON_DEPLOYMENT.md` §2.4), and ACT is a much larger graph. Paid
    # inside the loop, that lands as one command arriving SECONDS late with the
    # arm already energised, which is a safety problem and not a slow tick.
    #
    # The CPU build warms up too, for a smaller reason (allocation, first-touch
    # paging) and for a bigger one: the two numbers below are the whole basis
    # for choosing a device, so they have to be measured the same way on both.
    var warm_ms = 0.0
    var t_warm0 = perf_counter_ns()
    tr.predict(qpos_n, images_n, dummy_actions, dummy_valid, chunk)
    var first_ms = Float64(perf_counter_ns() - t_warm0) / 1e6
    for _ in range(WARMUP_QUERIES):
        var t_w = perf_counter_ns()
        tr.predict(qpos_n, images_n, dummy_actions, dummy_valid, chunk)
        warm_ms += Float64(perf_counter_ns() - t_w) / 1e6
    print(
        "            forward " + fixed(warm_ms / Float64(WARMUP_QUERIES), 1)
        + " ms warm (first " + fixed(first_ms, 1) + " ms, "
        + String(DEPLOY_TARGET) + ")"
    )
    if warm_ms / Float64(WARMUP_QUERIES) > 1000.0 / Float64(SO101_FPS) * 6.0:
        print(
            "   ⚠ a forward costs more than six 30 Hz steps. The arm will be"
            " driven mostly by\n     the ensemble's extrapolation between"
            " queries — read the `ensemble` line in the report."
        )

    # ── check 1: does THIS checkpoint go with THIS store? ─────────────────
    if check_steps > 0 and snap == "" and not have_store:
        print("")
        print(
            "── check 1 skipped: no --store, so no held-out episode to replay."
            " The statistics come from the policy's norm.json. ──"
        )
    if check_steps > 0 and snap == "" and have_store:
        ref ds = ds_opt.value()
        print("")
        print(
            "── replaying " + String(check_steps) + " steps of held-out"
            " episode " + String(ds.val_eps[0]) + " ──"
        )
        var ep = ds.val_eps[0]
        var ep_len = ds.store.episodes.length_of(ep)
        var n = check_steps if check_steps < ep_len else ep_len
        var g0 = ds.store.episodes.start_of(ep)
        var te0 = TemporalEnsemble[ADIM, K](m=ACT_TEMPORAL_ENSEMBLE_M)
        var s_act = 0.0
        var s_hold = 0.0
        var s_mean = 0.0
        var t_q0 = perf_counter_ns()
        for t in range(n):
            ds.fill_at[K](
                0, ep, t, qpos_n, images_n, dummy_actions, dummy_valid
            )
            tr.predict(qpos_n, images_n, dummy_actions, dummy_valid, chunk)
            te0.push(t, chunk, 0)
            te0.action_at(t, pred_n)
            denormalize(
                pred_n, 0, ds.action_mean, ds.action_std, pred, 0, ADIM
            )
            for j in range(ADIM):
                var truth = Float64(ds.action_raw[(g0 + t) * ADIM + j])
                s_act += abs(Float64(pred[j]) - truth)
                s_hold += abs(
                    Float64(ds.qpos_raw[(g0 + t) * QPOS + j]) - truth
                )
                s_mean += abs(Float64(ds.action_mean[j]) - truth)
        var ms_per_query = Float64(perf_counter_ns() - t_q0) / 1e6 / Float64(n)
        var d = Float64(n * ADIM)
        print(
            "   mean |error|   ACT " + fixed(s_act / d, 3)
            + "   hold " + fixed(s_hold / d, 3)
            + "   mean " + fixed(s_mean / d, 3)
            + "   (lerobot units)"
        )
        print("   " + fixed(ms_per_query, 1) + " ms per query")
        # ⚠ THE BAR IS THE CONSTANT BASELINE, NOT `hold`. Failing to beat
        # `hold` is a statement about the POLICY (see the open-loop
        # evaluator's note) and plenty of honest early policies do. Failing to
        # beat `mean` — predicting the dataset average — means the model and
        # these statistics do not belong together at all, which is the
        # wrong-store failure this check exists to catch.
        if s_act >= s_mean and not force:
            raise Error(
                "act deploy: the policy does not beat the constant `mean`"
                " baseline on its own held-out data. That is what deploying a"
                " checkpoint against the WRONG STORE looks like — the"
                " normalization statistics would be someone else's. Check"
                " --store, or pass --force if you know better."
            )
        if s_act >= s_hold:
            print(
                "   ⚠ ACT does not beat `hold` (predicting the current pose)."
                " The policy is weak, not mismatched — arming is allowed, but"
                " expect little."
            )

    # ── the cameras ───────────────────────────────────────────────────────
    print("")
    var names = norm.cameras.copy()
    var cams = List[CameraReader]()
    for i in range(N_CAM):
        var label = names[i] if i < len(names) else String("slot ") + String(i)
        print(
            "camera slot " + String(i) + " = " + pad_right(label, 26)
            + " <- " + devices[i]
        )
        # ⚠⚠ rgb=FALSE, AND out=IMG_W x IMG_H. Both matter, for one reason:
        # the camera thread now runs `camera_frame_to_chw_rgb`, which does the
        # resize AND the BGR->RGB swap in its transpose pass. Asking the
        # thread to swap as well (rgb=True, what `record.mojo` wants for its
        # encoder) would swap TWICE and feed the policy inverted colour
        # channels — a failure that looks like a bad policy, not like a bug.
        #
        # The resize moved here because that thread is blocked in `read()` for
        # most of a frame period anyway, and the 9.0 ms it took off the
        # control loop is what stood between 20 Hz and 30.
        var c = CameraReader.from_spec(
            devices[i], cam_w, cam_h, Float64(SO101_FPS), rgb=False,
            fourcc=cam_fourcc, out_w=IMG_W, out_h=IMG_H,
        )
        if undistort_dir.byte_length() > 0:
            if i >= len(names):
                raise Error(
                    "act deploy: --undistort needs the policy's camera names"
                    " (norm.json `cameras`) to find each slot's calibration"
                )
            var parts = names[i].split(".")
            var cp = undistort_dir + "/camera_" + String(parts[len(parts) - 1]) + ".txt"
            c.set_undistort(cp, UNDISTORT_FOVY)
            print("            undistorted to the sim pinhole via " + cp)
        # ⚠ 8 s, NOT THE 4 s DEFAULT. Measured on this rig: a camera that has
        # been idle takes longer than 4 s to report ready on its first open,
        # and `CameraReader.start` reports that as "device 0 did not report
        # ready" — which reads exactly like a camera that is not there.
        c.start(wait_ms=8000)
        # ⚠ SAY WHICH PIXEL FORMAT, because a format that differs from the
        # recording's is a silent train/deploy gap and nothing else reports it.
        var got = c.negotiated_fourcc()
        var where = c.resolved_node()
        var neg_fps = c.negotiated_fps()
        if where.byte_length() > 0 or got.byte_length() > 0 or neg_fps > 0.0:
            print(
                "            " + (where + "  " if where.byte_length() > 0
                                   else String(""))
                + ("format " + got + "  " if got.byte_length() > 0
                   else String(""))
                + (fixed(neg_fps, 1) + " fps" if neg_fps > 0.0
                   else String(""))
            )
        # ⚠ THE RATE IS A REQUEST LIKE THE SIZE. A camera that negotiated less
        # than the demonstrations were recorded at cannot feed the policy
        # observations at the cadence it learned, and nothing else says so.
        if neg_fps > 0.0 and neg_fps < Float64(SO101_FPS) - 1.0:
            print(
                "   ⚠ " + fixed(neg_fps, 1) + " fps, below the "
                + String(SO101_FPS) + " the demonstrations were recorded at."
            )
        cams.append(c^)
    print(
        "            " + String(cam_w) + "x" + String(cam_h) + " native ->"
        " " + String(IMG_W) + "x" + String(IMG_H) + " (PIL BILINEAR, the"
        " filter the store was built with)"
    )
    if cam_w != CAM_W or cam_h != CAM_H:
        print(
            "   ⚠⚠ NOT the " + String(CAM_W) + "x" + String(CAM_H)
            + " the demonstrations were recorded at. The resize filter"
            " depends on the reduction factor, so these are different pixels"
            " from the same scene."
        )

    # ── --snap: is physical camera i the camera that filled slot i? ───────
    #
    # ⚠⚠ NOTHING ELSE CAN CATCH A SWAP. Two cameras of the same resolution
    # produce a perfectly well-formed observation either way round; the model
    # simply sees a world it has never been shown, and the failure presents as
    # "the policy does not work" rather than as an error. Device indices are
    # not stable either — they are whatever order the USB stack enumerated in,
    # and this machine currently has THREE cameras attached where the
    # recording had two.
    #
    # So: write what the STORE holds in each slot beside what each device is
    # showing right now, at the same size and through the same resize, and let
    # the operator look. It is the only honest test.
    if snap != "":
        # ⚠ ONE BUFFER, NOT TWO. The camera thread now delivers the CHW RGB
        # tensor itself, so there is no native-size frame to hold on to here.
        var snap_chw = List[UInt8](length=CAM_ELEMS, fill=0)
        var row = List[Scalar[DType.uint8]]()
        if snap_from == "":
            ref ds = ds_opt.value()
            ds.image_row_u8(ds.store.episodes.start_of(ds.val_eps[0]), row)
        else:
            # ⚠ ffmpeg WILL SHOUT "Broken pipe" HERE, AND IT IS EXPECTED.
            # Only the FIRST frame of each episode is wanted, so the decoder
            # is closed while it still has a video to write; it gets EPIPE on
            # the next packet and says so on stderr before exiting. Its stderr
            # is deliberately NOT silenced — a genuinely corrupt video
            # announces itself the same way, and one of those is worth
            # hearing.
            print(
                "  reading the first frame of each episode from " + snap_from
                + "\n  (ffmpeg prints 'Broken pipe' below — expected: we"
                " close it after one frame)"
            )
            row = recorded_first_row[N_CAM, IMG_H, IMG_W](snap_from, names)
        var hwc = List[UInt8](length=IMG_W * IMG_H * 3, fill=0)
        for i in range(N_CAM):
            var label = (
                names[i] if i < len(names) else String("slot") + String(i)
            )
            # the store's slot i, CHW -> HWC
            for p in range(IMG_W * IMG_H):
                hwc[p * 3] = UInt8(row[i * CAM_ELEMS + p])
                hwc[p * 3 + 1] = UInt8(
                    row[i * CAM_ELEMS + IMG_W * IMG_H + p]
                )
                hwc[p * 3 + 2] = UInt8(
                    row[i * CAM_ELEMS + 2 * IMG_W * IMG_H + p]
                )
            var p_store = snap + "/slot" + String(i) + "_store.png"
            save_png(p_store, hwc, IMG_W, IMG_H, 3)

            # device i, right now, through the SAME resize the policy sees
            if not cams[i].take_blocking(snap_chw):
                raise Error(
                    "act deploy: camera " + devices[i]
                    + " delivered no frame for --snap"
                )
            for p in range(IMG_W * IMG_H):
                hwc[p * 3] = snap_chw[p]
                hwc[p * 3 + 1] = snap_chw[IMG_W * IMG_H + p]
                hwc[p * 3 + 2] = snap_chw[2 * IMG_W * IMG_H + p]
            var p_live = snap + "/slot" + String(i) + "_live.png"
            save_png(p_live, hwc, IMG_W, IMG_H, 3)
            print(
                "  " + pad_right(label, 26) + " store -> " + p_store
                + "   live -> " + p_live
            )

            # ⚠⚠ AND THE NUMBERS, BECAUSE "COMPARE THE PAIR" IS NOT A TEST.
            # Two PNGs on a headless board are a request for a human to
            # eyeball a brightness shift over SSH, and a systematic one is
            # exactly what the eye is worst at — a policy fed images 20%
            # darker than it trained on has no way to say so, and the failure
            # arrives as "the policy transfers badly".
            #
            # Per-channel means over the SAME tensor the policy consumes,
            # after the same resize, so this is the actual input distribution
            # and not a property of the PNG.
            var s_sum = List[Float64](length=3, fill=0.0)
            var l_sum = List[Float64](length=3, fill=0.0)
            var n_px = Float64(IMG_W * IMG_H)
            for c in range(3):
                for p in range(IMG_W * IMG_H):
                    s_sum[c] += Float64(
                        Int(row[i * CAM_ELEMS + c * IMG_W * IMG_H + p])
                    )
                    l_sum[c] += Float64(Int(snap_chw[c * IMG_W * IMG_H + p]))
            var s_mean = (s_sum[0] + s_sum[1] + s_sum[2]) / (3.0 * n_px)
            var l_mean = (l_sum[0] + l_sum[1] + l_sum[2]) / (3.0 * n_px)
            print(
                "      mean pixel  store " + fixed(s_mean, 1)
                + "   live " + fixed(l_mean, 1)
                + "   ratio " + fixed(l_mean / s_mean if s_mean > 0.0 else 0.0, 2)
                + "   (R " + fixed(s_sum[0] / n_px, 0) + "/"
                + fixed(l_sum[0] / n_px, 0) + "  G "
                + fixed(s_sum[1] / n_px, 0) + "/" + fixed(l_sum[1] / n_px, 0)
                + "  B " + fixed(s_sum[2] / n_px, 0) + "/"
                + fixed(l_sum[2] / n_px, 0) + ")"
            )
            # ⚠ A RATIO, NOT A DIFFERENCE, and no threshold on the scene
            # content: the cube and the arm move, so the images SHOULD differ.
            # What must not differ is the overall exposure.
            var ratio = l_mean / s_mean if s_mean > 0.0 else 1.0
            if ratio < 0.8 or ratio > 1.25:
                print(
                    "      ⚠⚠ the live view is " + fixed(ratio, 2)
                    + "x the recorded brightness. That is a train/deploy gap"
                    " in the OBSERVATION,\n         not in the policy — fix"
                    " the lighting or the exposure before reading anything"
                    " into how the arm behaves."
                )
        for i in range(N_CAM):
            try:
                cams[i].stop()
            except:
                pass
        print("")
        print(
            "compare each pair. If slot 0's store frame is the SIDE view and"
            " device 0 is looking\nfrom the front, swap --devices before"
            " arming anything."
        )
        return

    # ── the arm ───────────────────────────────────────────────────────────
    print("")
    var f_port = follower_port(port_arg)
    print("follower    " + f_port)
    var why_port = port_refusal(f_port, String("follower"))
    if why_port.byte_length() > 0:
        raise Error("act deploy: " + why_port)
    var follower = SO101Arm(
        f_port,
        max_step_ticks=step_ticks,
        track_step_ticks=TRACK_STEP_TICKS,
    )
    follower.bus.timeout_ms = 20

    var raw = Array[Int32, SO101_N](fill=0)
    if follower.read_positions(Span(raw)) != SO101_N:
        raise Error(
            "act deploy: the follower did not report " + String(SO101_N)
            + " positions — not arming"
        )

    # ── check 2: is the arm anywhere the demonstrations went? ─────────────
    print("")
    print("   joint          present   demonstrated range        action clamp")
    var outside = 0
    for i in range(SO101_N):
        var p = follower.cal.degrees(i, raw[i])
        var note = String("")
        if p < q_lo[i] or p > q_hi[i]:
            outside += 1
            note = " ⚠ OUTSIDE"
        print(
            "   " + pad_right(joint_name(i), 14)
            + col(p, 8, 1) + "   [" + col(q_lo[i], 7, 1) + ","
            + col(q_hi[i], 7, 1) + " ]   [" + col(a_lo[i], 7, 1) + ","
            + col(a_hi[i], 7, 1) + " ]" + note
        )
    if outside > 0:
        print(
            "   ⚠ " + String(outside) + " joint(s) start outside the"
            " demonstrated pose box. The policy has never seen this"
            " observation;\n     its first command is an extrapolation and"
            " the step clamp is what makes that survivable."
        )

    # ── go ────────────────────────────────────────────────────────────────
    var stdin = StdinReader()
    var interactive = stdin_is_tty()
    print("")
    if arm_it:
        print(
            "⚠⚠ THE FOLLOWER WILL BE ENERGISED AND WILL MOVE FOR "
            + String(seconds) + " s."
        )
    else:
        print("dry run — torque stays OFF and the arm is backdrivable.")
    if interactive:
        print("press Enter to start (q = quit), and Enter again to stop early")
    else:
        print(
            "press Enter to start (q = quit). stdin is not a terminal, so the"
            " run ends on its\n  --seconds and Enter-to-stop is off."
        )
    stdin.discard_pending()
    var answer = stdin.line()
    if answer == "q" or answer == "Q":
        for i in range(N_CAM):
            try:
                cams[i].stop()
            except:
                pass
        print("nothing was armed.")
        return

    # ⚠ THE POSE TO COME BACK TO, captured BEFORE anything is energised. The
    # arm was resting here under gravity alone, which is what makes it the one
    # pose we know is safe to release torque at. `raw` is overwritten every
    # tick of the loop below, so it has to be copied now.
    var start_pose = List[Int32](length=SO101_N, fill=0)
    for i in range(SO101_N):
        start_pose[i] = raw[i]

    if arm_it:
        # Guard 4: park the goal on the present pose BEFORE torque, so arming
        # holds instead of snapping to a stale `Goal_Position`.
        follower.set_position_mode()
        var hold = follower.max_step_ticks
        follower.max_step_ticks = 0
        follower.write_goals(Span(raw))
        follower.max_step_ticks = hold
        follower.set_torque(True)
        print("follower torque ON\n")
    else:
        print("dry run — nothing energised\n")

    # ⚠ `frames[i]` IS the CHW tensor now (the camera thread resized it), so
    # the separate staging buffer this loop used to transpose into is gone.
    var frames = List[List[UInt8]]()
    for i in range(N_CAM):
        frames.append(
            List[UInt8](unsafe_uninit_length = cams[i].frame_bytes())
        )
    var goals = Array[Int32, SO101_N](fill=0)
    var cmd = List[Float64](length=ADIM, fill=0.0)
    for i in range(ADIM):
        cmd[i] = follower.cal.degrees(i, raw[i])

    var te = TemporalEnsemble[ADIM, K](m=ACT_TEMPORAL_ENSEMBLE_M)

    var queries = 0
    var bus_skipped = 0
    var refused = 0
    var clamped = 0
    var stale_ticks = 0
    var reversals = 0.0
    var n_step = 0.0
    var sum_step = 0.0
    var max_step_seen = 0.0
    var sum_contrib = 0
    var worst_cam = 0.0
    var worst_pre = 0.0
    var worst_fwd = 0.0
    var worst_bus = 0.0
    var sum_cam = 0.0
    var sum_pre = 0.0
    var sum_fwd = 0.0
    var sum_bus = 0.0
    var last_delta = List[Float64](length=SO101_N, fill=0.0)
    var last_goal = List[Float64](length=SO101_N, fill=0.0)
    for i in range(SO101_N):
        last_goal[i] = Float64(raw[i])
    var last_t_cmd = -1

    # ⚠⚠ THE FRAME COUNTERS ARE SAMPLED HERE, NOT READ AS TOTALS. The camera
    # threads started at `c.start()` — before check 2, before the arm section,
    # and before the operator pressed Enter — so a total divided by the LOOP's
    # elapsed time reports the frames of a minute of idling as if they had
    # arrived during the run. The first board run printed "95.2 fps delivered
    # (negotiated 30.0)", which is not a camera that exists.
    var loop_ns = 0
    var frames_at_start = List[Int]()
    var frames_at_end = List[Int]()
    for i in range(N_CAM):
        frames_at_start.append(cams[i].frames_delivered())
    var loop_t0 = perf_counter_ns()
    var deadline = loop_t0 + seconds * 1_000_000_000
    try:
        while perf_counter_ns() < deadline:
            # ⚠ ONLY ON A TERMINAL. At the end of a pipe `poll` reports
            # POLLHUP as readiness and `line()` hands back the same empty
            # string an Enter does, so a scripted `printf '\\n' | act_deploy`
            # would stop on its very first check. See `stdin_is_tty`.
            if interactive and stdin.has_input():
                _ = stdin.line()
                print("  stopped by the operator")
                break

            # ── observe ──────────────────────────────────────────────────
            var t_c0 = perf_counter_ns()
            for i in range(N_CAM):
                # ⚠ THE NEWEST FRAME, NOT THE OLDEST — see `take_latest`. A
                # forward takes ~3 frame periods, so the ring always holds a
                # queue and taking its head would add 100 ms of staleness to
                # every observation. `take_blocking` only ever runs on the
                # first tick, before the cameras have queued anything.
                if cams[i].take_latest(frames[i]) == 0:
                    stale_ticks += 1
                    if not cams[i].take_blocking(frames[i]):
                        raise Error(
                            "act deploy: camera " + devices[i]
                            + " stopped delivering frames"
                        )
            var t_c1 = perf_counter_ns()

            # The grid step this observation BELONGS to, on the 30 Hz clock
            # the demonstrations were recorded on.
            var t_obs = Int(
                Float64(t_c1 - loop_t0) * Float64(SO101_FPS) / 1e9
            )
            # ⚠ STRICTLY INCREASING, and past the last step already COMMANDED.
            # The ensemble's ring is indexed by query step (`slot = t % K`), so
            # a query landing on a step that has already been acted on would
            # be a chunk whose first entries describe the past. Forcing it
            # forward costs at most one grid step of accuracy in `t_obs` and
            # keeps every chunk in the ring speaking about the future.
            if t_obs <= last_t_cmd:
                t_obs = last_t_cmd + 1

            var got = follower.read_positions(Span(raw))
            if got != SO101_N:
                # Guard 5: a partial read is not an observation. Skipping the
                # whole tick holds the last goal, which is safe; a
                # half-updated pose fed to the policy is not.
                bus_skipped += 1
                continue
            for j in range(QPOS):
                qpos_n[j] = (
                    Scalar[DT](follower.cal.degrees(j, raw[j]))
                    - norm.qpos_mean[j]
                ) / norm.qpos_std[j]

            for i in range(N_CAM):
                # ⚠ NO RESIZE HERE ANY MORE — `frames[i]` IS the CHW RGB
                # tensor, produced on the camera thread. What is left is the
                # normalizer, which is cheap and must stay: it is the same
                # call `ACTDataset._fill_one` makes, and the one step that has
                # to agree exactly between training and deployment.
                normalize_camera_chw[IMG_H, IMG_W](
                    frames[i], 0, images_n, i * CAM_ELEMS
                )
            var t_pre = perf_counter_ns()

            # ── think ────────────────────────────────────────────────────
            tr.predict(qpos_n, images_n, dummy_actions, dummy_valid, chunk)
            te.push(t_obs, chunk, 0)
            queries += 1
            var t_fwd = perf_counter_ns()

            # ── act, at the grid step it is NOW ──────────────────────────
            # ⚠⚠ NOT `action_at(t_obs)`. The forward took ~3 grid steps; the
            # arm is being commanded now, so the action it gets must be the
            # one the trajectory calls for NOW. Commanding `t_obs` would run
            # the whole demonstration permanently one latency behind itself.
            var t_cmd = Int(
                Float64(t_fwd - loop_t0) * Float64(SO101_FPS) / 1e9
            )
            if t_cmd < t_obs:
                t_cmd = t_obs
            if t_cmd > t_obs + K - 1:
                # A stall longer than a whole chunk. The freshest query cannot
                # speak about this step, so ask for the last one it can.
                t_cmd = t_obs + K - 1
            last_t_cmd = t_cmd
            sum_contrib += te.n_contributors(t_cmd)
            te.action_at(t_cmd, pred_n)
            denormalize(
                pred_n, 0, norm.action_mean, norm.action_std, pred, 0, ADIM
            )

            for j in range(ADIM):
                var v = Float64(pred[j])
                # Guard 6: the dataset's own action box.
                if v < a_lo[j]:
                    v = a_lo[j]
                    clamped += 1
                elif v > a_hi[j]:
                    v = a_hi[j]
                    clamped += 1
                # ⚠ OFF BY DEFAULT (`--smooth 1.0`). Temporal ensembling IS
                # the smoother here — every command already blends ~20
                # overlapping chunks — and `deploy_reach_real.mojo`'s EMA
                # exists because a SAC policy trained without an action-rate
                # penalty chattered. Reach for it only if `reversals` in the
                # report says the command is buzzing.
                cmd[j] = (1.0 - smooth) * cmd[j] + smooth * v
                goals[j] = follower.cal.raw_from_degrees(j, cmd[j])

                var delta = Float64(goals[j]) - last_goal[j]
                if delta * last_delta[j] < 0.0:
                    reversals += 1.0
                last_delta[j] = delta
                last_goal[j] = Float64(goals[j])
                var stp = Float64(goals[j] - raw[j])
                if stp < 0.0:
                    stp = -stp
                if stp > max_step_seen:
                    max_step_seen = stp
                sum_step += stp
                n_step += 1.0

            if arm_it:
                try:
                    follower.write_goals(Span(goals))
                except:
                    refused += 1
            var t_bus = perf_counter_ns()

            var d_cam = Float64(t_c1 - t_c0) / 1e6
            var d_pre = Float64(t_pre - t_c1) / 1e6
            var d_fwd = Float64(t_fwd - t_pre) / 1e6
            var d_bus = Float64(t_bus - t_fwd) / 1e6
            if d_cam > worst_cam:
                worst_cam = d_cam
            if d_pre > worst_pre:
                worst_pre = d_pre
            if d_fwd > worst_fwd:
                worst_fwd = d_fwd
            if d_bus > worst_bus:
                worst_bus = d_bus
            sum_cam += d_cam
            sum_pre += d_pre
            sum_fwd += d_fwd
            sum_bus += d_bus

            if queries % 10 == 0:
                var line = String("  t=") + pad_left(
                    fixed(Float64(t_cmd) / Float64(SO101_FPS), 1), 5
                ) + "s  x" + pad_left(
                    String(te.n_contributors(t_cmd)), 3
                ) + " "
                for j in range(ADIM):
                    line += " " + col(cmd[j], 7, 1)
                print(line)
    finally:
        # ⚠⚠ THE CLOCK STOPS HERE, NOT AFTER THE SHUTDOWN. Everything below —
        # the ramp back to the start pose, and an operator taking their time
        # over "press Enter to release torque" — is not part of the control
        # loop, and counting it reported a 10 s run at 30 Hz as "301 queries
        # in 24.6 s = 12.2 Hz". A rate is only a rate over the interval that
        # produced it.
        #
        # ⚠ IN THE `finally`, so an exception mid-run still gets an honest
        # elapsed rather than one that includes the unwind.
        #
        # ⚠⚠ MOJO WARNS "assignment to 'loop_ns' was never used" HERE AND ON
        # ITS DECLARATION. THE WARNING IS WRONG — do not "fix" it by deleting
        # the assignment or by `_ =`, which would leave `elapsed` reading the
        # initial 0 and report `inf Hz`. Reduced to eight lines and checked:
        #
        #     var v = 0
        #     try: work()
        #     finally: v = 42
        #     print(v)          # prints 42, and warns twice
        #
        # The compiler's dataflow does not see a `finally` assignment reaching
        # the code after the statement; the generated code does.
        loop_ns = perf_counter_ns() - loop_t0
        # ⚠ AND THE FRAME COUNTS WITH IT, at the same instant. The cameras
        # keep running through the ramp and the Enter wait, so a count read
        # after them over an elapsed measured before them is two different
        # intervals in one ratio — the numerator and the denominator have to
        # end together.
        for i in range(N_CAM):
            frames_at_end.append(cams[i].frames_delivered())
        # ⚠ THE SAME SHUTDOWN ON EVERY PATH, including an exception. A camera
        # that stops delivering mid-run leaves the arm extended, and the old
        # code's unconditional `set_torque(False)` would drop it there — the
        # error path is precisely when a fall is least expected. `--no-return`
        # is the only way to get the old behaviour.
        var released = return_and_release(
            follower, start_pose, arm_it, do_return, stdin, interactive
        )
        if not released:
            print(
                "⚠ the follower is STILL ENERGISED — that is deliberate, see"
                " above."
            )
        for i in range(N_CAM):
            try:
                cams[i].stop()
            except:
                pass

    var elapsed = Float64(loop_ns) / 1e9
    print("=" * 74)
    print("ACT closed-loop run")
    print("  queries           = " + String(queries) + " in "
          + fixed(elapsed, 1) + " s = "
          + fixed(Float64(queries) / elapsed, 1) + " Hz")
    print(
        "  action grid       = " + String(SO101_FPS) + " Hz, so the policy"
        " got a waypoint every "
        + (fixed(Float64(SO101_FPS) * elapsed / Float64(queries), 1)
           if queries > 0 else String("n/a"))
        + " demonstrated steps"
    )
    print(
        "  ensemble          = "
        + (fixed(Float64(sum_contrib) / Float64(queries), 1)
           if queries > 0 else String("n/a"))
        + " chunks per command (K=" + String(K) + " is the ceiling)"
    )
    print("  bus-skipped ticks = " + String(bus_skipped))
    print("  write refused     = " + String(refused))
    # ⚠ STARVED IS NOT AUTOMATICALLY BAD — it means the ring was empty, so
    # the loop is FASTER than the camera and the frame it then waits for is
    # maximally fresh. The failure it would hide is the opposite one, and the
    # delivered rate is what tells them apart: a camera meeting its negotiated
    # rate starves a faster consumer honestly; one below it is the bottleneck.
    print(
        "  camera starved    = " + String(stale_ticks) + " of "
        + String(queries * N_CAM) + " frame takes"
    )
    for i in range(N_CAM):
        var delivered = (
            frames_at_end[i] - frames_at_start[i]
            if i < len(frames_at_end) else 0
        )
        var rate = Float64(delivered) / elapsed if elapsed > 0.0 else 0.0
        var claimed = cams[i].negotiated_fps()
        print(
            "     camera " + String(i) + "      " + pad_left(String(delivered), 5)
            + " frames = " + fixed(rate, 1) + " fps delivered  (negotiated "
            + fixed(claimed, 1) + ")"
        )
        # ⚠ V4L2 REPORTS CAP_PROP_FPS AS WHAT WAS ASKED FOR, not always as what
        # it does. The delivered count is the only one of the two that was
        # measured, so when they disagree it is the claim that is wrong.
        if claimed > 0.0 and rate > claimed * 1.2:
            print(
                "       ⚠ delivering " + fixed(rate / claimed, 1) + "x its"
                " claimed rate — the extra frames are decoded and dropped,"
                " and cost USB bandwidth and CPU for nothing."
            )
        # ⚠⚠ THE SLOW CAMERA IS THE LOOP'S SPEED LIMIT, and it is invisible
        # in every other line of this report. A camera below its negotiated
        # rate PACES the control loop: the loop waits for it, so work taken
        # off the control thread is absorbed by a longer wait and the query
        # rate does not move at all. Measured exactly that way on the board —
        # 5.3 ms saved in preprocess, 5.3 ms added to the camera wait, 20.0 Hz
        # before and after.
        #
        # The usual cause is AUTO-EXPOSURE in a dim scene: a USB camera
        # lengthens its exposure and silently halves its frame rate. Two
        # identical cameras disagreeing is the tell — it is the light, not the
        # hardware.
        elif claimed > 0.0 and rate < claimed * 0.9:
            print(
                "       ⚠⚠ " + fixed(rate, 1) + " fps against " + fixed(claimed, 1)
                + " negotiated — THIS CAMERA PACES THE LOOP. Work taken off"
                " the control thread\n          cannot help while it is the"
                " constraint. Usual cause: auto-exposure in a dim scene, which"
                "\n          this control turns off — and it does NOT survive"
                " a replug or a reboot:\n"
                "            v4l2-ctl -d " + cams[i].resolved_node()
                + " --set-ctrl=exposure_dynamic_framerate=0\n"
                "          ⚠ It also SHORTENS the exposure, so the pixels get"
                " darker. If the demonstrations\n          were recorded"
                " without it, that is a train/deploy shift — check the frames,"
                " or\n          light the scene instead. `v4l2-ctl -d "
                + cams[i].resolved_node() + " --all` shows the current value."
            )
    # ⚠ THE OFF-DISTRIBUTION SIGNAL. Every one of these is the policy asking
    # for a pose no demonstration ever reached.
    print(
        "  action clamped    = " + String(clamped) + " of "
        + String(Int(n_step)) + " joint-commands"
        + ("   ⚠ the policy is asking for poses the demonstrations never"
           " reached" if clamped * 20 > Int(n_step) else "")
    )
    # ⚠ IN A DRY RUN THIS IS NOT A PER-TICK DEMAND. It is |commanded −
    # present|, and in a dry run the arm never moves, so it settles at the
    # STANDING DISTANCE between where the arm is parked and where the policy
    # wants it. Only an armed run makes it the quantity `max_step_ticks`
    # bounds.
    print(
        "  commanded step    = mean "
        + (fixed(sum_step / n_step, 1) if n_step > 0 else String("n/a"))
        + " ticks, max " + fixed(max_step_seen, 1)
        + " (clamp is " + String(step_ticks) + ")"
        + ("   <- standing distance, the arm never moved"
           if not arm_it else "")
    )
    print(
        "  goal reversals    = "
        + (fixed(100.0 * reversals / n_step, 0) if n_step > 0
           else String("n/a"))
        + " % of writes changed DIRECTION  <- this is the shake"
    )
    var throttle = (sum_step / n_step) / Float64(step_ticks) if (
        n_step > 0 and step_ticks > 0
    ) else 0.0
    if throttle > 2.0 and arm_it:
        print(
            "  ⚠ RATE-LIMITED: the policy asked for " + fixed(throttle, 1)
            + "x the clamp on average.\n     The CLAMP shaped this run, not"
            " the policy — raise --step before reading anything into the"
            " motion."
        )
    # ⚠ MEAN BESIDE WORST, because they say different things and only the
    # mean sets the query rate. A worst-case `forward` of 250 ms against a
    # mean of 100 ms is one scheduling hiccup, not the cost of the model — and
    # optimising the wrong one of those is a wasted afternoon.
    var qd = Float64(queries) if queries > 0 else 1.0
    print("  per query, mean (worst) in ms")
    print(
        "     cameras    " + pad_left(fixed(sum_cam / qd, 1), 6)
        + " (" + fixed(worst_cam, 1) + ")"
        + "      preprocess " + pad_left(fixed(sum_pre / qd, 1), 6)
        + " (" + fixed(worst_pre, 1) + ")"
    )
    print(
        "     forward    " + pad_left(fixed(sum_fwd / qd, 1), 6)
        + " (" + fixed(worst_fwd, 1) + ")"
        + "      bus        " + pad_left(fixed(sum_bus / qd, 1), 6)
        + " (" + fixed(worst_bus, 1) + ")"
    )
    if not arm_it:
        print("  ⚠ DRY RUN — nothing was written to the arm. Add --arm.")
    print("=" * 74)
