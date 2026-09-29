# +--------------------------------------------------------------------------+ #
# | A SIM-TRAINED PIXEL STUDENT on the PHYSICAL SO-101 — the sim-to-real test
# +--------------------------------------------------------------------------+ #
"""Drive the follower from a camera policy trained ENTIRELY IN SIMULATION
(`noeira/tasks/pixel_dagger_tower.mojo`: DAgger from a state PPO teacher),
observing the rig through its two real cameras.

    pixi run build-opencv                     # ONCE
    pixi run build-serial                     # ONCE

    # the policy as a ROLE (promoted with tools/project/promote_pixel_student.sh)
    # ON THE JETSON — SAFE BY DEFAULT: reads the arm and the cameras, runs the
    # policy, prints every command it WOULD send, energises nothing:
    pixi run -e jetson pixel-deploy-jetson -- --role pixel_lift --snap /tmp/px_snap

    # --arm is what moves the robot. Be at the desk, hand on the power.
    pixi run -e jetson pixel-deploy-jetson -- --role pixel_lift --arm --seconds 20

    # on the Mac (CPU, the cameras by index):
    pixi run mojo build -I . -Xlinker -ld_classic -o /tmp/px_deploy \\
        examples/so101/pixel_student_deploy_real.mojo
    /tmp/px_deploy --role pixel_lift --devices 0,1

Flags: --project (so101-tower), --role (pixel_lift) or --ckpt PATH, --devices
CSV (overhead first), --port, --fourcc, --undistort DIR (projects/so101-tower/
cameras), --arm, --seconds (20), --no-return, --no-start-pose, --snap DIR,
--gripper-sign 0|1 (default: the manifest's), --step-ticks (80), --force-dark,
--record DIR (per-tick CSV of q, qd, action, target + the two frames every
16 ticks — what a run did, to lay beside `pixel_student_probe_sim.mojo`),
--grip-offset X (rad, default 0: added to the GRIPPER angle the policy SEES,
not to the one the commands use — the real jaws read ~0.05 rad more closed
on the 25 mm cube than the sim's, 0.063 vs 0.10-0.14, and a student never
trained on that reading hovered over the bowl without releasing),
--sysid FILE (with --arm: NO policy — each joint in turn steps +A, back, -A,
back from the sim's start pose, 0.8 s per step, A 0.1 rad / 0.3 gripper; the
per-tick targets and joints go to FILE, the servos' delay and time constant
are fitted from it (`delta_action.ServoLag`'s numbers)).

## What makes a SIM policy's observation on the real rig (`pixel_student.mojo`)

1. each camera frame is UNDISTORTED to the simulator's pinhole (fisheye
   calibration `camera_<name>.txt`, fovy 73.7398 deg) at 640 x 480 — the
   sim cameras are pinholes, the real ones are not;
2. the centred 480 x 480 square is kept — the sim traces a SQUARE picture at
   that fovy in both directions — and averaged down to the student's 16 x 16
   (`frame_to_planes`), RGB / 255 - 0.5; slot 0 overhead, slot 1 wrist;
3. the six joints come from the follower in MODEL RADIANS through the rig's
   measured joint zero (`SimJointMap.tower_follower`, the map
   `tower_expert_real.mojo` drives this arm with).

The action is the teacher's: target = clamp(q + a x 0.05 rad, gripper 0.2),
`delta_action.delta_target` — the SAME function the policy was trained
through — then `from_sim` to servo ticks. The loop runs at the SIM's control
period (32 ms, 16 x 2 ms: the manifest's `control_period_s`), spin-paced; a
camera frame older than one period is reused and counted.

⚠⚠ **THE POLICY HAS NEVER SEEN A REAL IMAGE.** It was trained on the rig's
CALIBRATED LOOK (lights and albedos fitted to recorded frames) with no domain
randomisation yet. `--snap DIR` writes, before anything moves, each camera's
undistorted frame, its square crop and the exact 16 x 16 the policy receives
— compare them with the sim's (`dagger_tower_pixels --png`) before an armed
run. A failure here is most likely THE PICTURES; that is what the snap is for.

⚠ THE EPISODE STARTS WHERE THE SIM'S DID. Sim episodes start at the family's
reset pose; a policy started elsewhere is off its training distribution from
step 0. When armed, the follower first ramps (slowly, under the return clamp)
to the sim's start pose; `--no-start-pose` skips that. Place the brick where
the task's placements put it (cube-in-bowl's recorded layouts: the desk in
front of the arm, 15-40 cm out — the student is weakest under 20 cm).

## Safety — the ACT deploy's rules, unchanged

`--arm` is OFF by default (dry run: torque stays off, every would-be command
printed). Goals parked on the present pose before torque; `SO101Arm`'s step
clamp (80 ticks, then tracking); a partial bus read skips the tick's write;
targets clamped to the MODEL's joint limits; the map's interior round-trip
checked before arming; the run ends with `return_and_release` — back to the
pose the run started from, torque left ON if it does not arrive. ⚠ A `finally`
does not cover an abort or a signal: the recovery is `pixi run
soarm-torque-off` and the power switch.
"""

from std.os import makedirs
from std.os.path import dirname, exists
from std.sys import argv
from std.time import perf_counter_ns

from noeira.core.policy import describe_policy, resolve_policy
from noeira.io.fileio import StdinReader, stdin_is_tty
from noeira.io.png import save_png
from noeira.nn.constants import DT
from noeira.nn.core.checkpoint import load_params
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.robot.so101 import SO101Arm, SO101_N, joint_name
from noeira.robot.so101.deploy_shutdown import (
    RETURN_STEP_TICKS, RETURN_TOLERANCE_TICKS, return_and_release,
)
from noeira.robot.so101.ports import follower_port, port_refusal
from noeira.robot.so101.sim_map import SimJointMap
from noeira.tasks.delta_action import delta_target
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TowerConfig
from noeira.tasks.pixel_student import (
    StudentNet, N_CAMS, OBS_PX, PLANE, IN_DIM, ACT, CAM_FOVY_DEG, JOINT_VEL,
    camera_names, check_pixel_manifest, frame_to_planes, joints_to_planes,
    joint_vels_to_planes, student_act, HIST_WORDS, act_hist_push,
    act_hist_to_planes,
)
from noeira.tasks.placement.so101_tower import So101TowerPlacement
from noeira.tasks.posed_reset import posed_qpos
from noeira.tasks.spec import load_family
from noeira.utils.fmt import col, fixed, pad_left, pad_right
from noeira.vision.camera_thread import CameraReader, parse_camera_specs

comptime FAMILY = "so101_tower"
comptime FAMILY_PATH = "noeira/tasks/families/so101_tower.family"
comptime CAM_W = 640
comptime CAM_H = 480
comptime MAX_STEP_TICKS = 80
comptime TRACK_STEP_TICKS = 512
comptime START_POSE_TIMEOUT_S = 10
comptime CAMERA_WARMUP_S = 3
"""⚠ THE FIRST FRAMES OF A UVC CAMERA ARE DARK AND GREEN — auto-exposure and
white balance have not settled. The first bring-up (27 Sep) snapped the very
first frame: mean RGB (3, 18, 4) against the sim's ~146 grey, and the policy
answered with actions of 4 and 8 on a [-1, 1] scale. Frames are drained for
this long before anything is observed."""
comptime SIM_MEAN_OVERHEAD = 0.571
comptime SIM_MEAN_WRIST = 0.600
"""The sim's mean pixel (0..1) of each camera's policy view at the task's
reset (`dagger_tower_pixels --png`, lift on the real layouts, 27 Sep). A real
view darker than HALF of it refuses `--arm` (`--force-dark` overrides): a
network fed a picture that far from its training set acts with confidence
and without meaning."""


def _arg(args: List[String], key: String, default: String) -> String:
    for i in range(len(args) - 1):
        if args[i] == key:
            return args[i + 1]
    return default


def _flag(args: List[String], key: String) -> Bool:
    for a in args:
        if a == key:
            return True
    return False


def _manifest_beside(ckpt: String) -> String:
    """`policies/<role>.ckpt` -> `policies/<role>.norm.json`; a run's
    `checkpoints/last.ckpt` -> `checkpoints/norm.json`."""
    if ckpt.endswith(".ckpt"):
        var p = String(ckpt[byte = 0 : ckpt.byte_length() - 5]) + ".norm.json"
        if exists(p):
            return p
    return dirname(ckpt) + "/norm.json"


def _spin_until(t: Int):
    while perf_counter_ns() < t:
        pass


def _ramp_to(mut arm: SO101Arm, ref target: List[Int32], timeout_s: Int) -> Bool:
    """Ramp the energised follower to `target` under the RETURN clamp (the
    slow one), confirming arrival. The mirror of `return_and_release`'s ramp,
    toward the sim's start pose instead of the resting one."""
    var hold = arm.max_step_ticks
    var hold_track = arm.track_step_ticks
    arm.max_step_ticks = RETURN_STEP_TICKS
    arm.track_step_ticks = 0
    var goals = Array[Int32, SO101_N](fill=0)
    for i in range(SO101_N):
        goals[i] = target[i]
    var present = Array[Int32, SO101_N](fill=0)
    var t_end = perf_counter_ns() + timeout_s * 1_000_000_000
    var arrived = False
    while perf_counter_ns() < t_end:
        var t0 = perf_counter_ns()
        try:
            arm.write_goals(Span(goals))
            if arm.read_positions(Span(present)) == SO101_N:
                var worst = 0
                for i in range(SO101_N):
                    var d = Int(present[i]) - Int(target[i])
                    if d < 0:
                        d = -d
                    if d > worst:
                        worst = d
                if worst <= RETURN_TOLERANCE_TICKS:
                    arrived = True
                    break
        except:
            break
        _spin_until(t0 + 1_000_000_000 // 30)
    arm.max_step_ticks = hold
    arm.track_step_ticks = hold_track
    return arrived


def _view_mean(ref x: List[Scalar[DT]], cam: Int) -> Float64:
    """Mean pixel (0..1) of camera `cam`'s planes in the policy input."""
    var acc = 0.0
    for k in range(3 * PLANE):
        acc += Float64(x[3 * cam * PLANE + k]) + 0.5
    return acc / Float64(3 * PLANE)


def _warm_cameras(
    mut cams: List[CameraReader], mut frames: List[List[UInt8]], seconds: Int
) raises:
    """Drain frames for `seconds` so exposure and white balance settle."""
    var t_end = perf_counter_ns() + seconds * 1_000_000_000
    var n = 0
    while perf_counter_ns() < t_end:
        for i in range(len(cams)):
            if cams[i].take_blocking(frames[i], timeout_ms=1000):
                n += 1
    print("  cameras warmed for " + String(seconds) + " s (" + String(n)
          + " frames drained)")


def _snap(
    dir: String, ref frames: List[List[UInt8]], ref x: List[Scalar[DT]],
    ref q: List[Float64], tag: String = "",
) raises:
    """Each camera: the undistorted frame, and the 16x16 the policy receives
    (upscaled x16), as PNGs."""
    makedirs(dir, exist_ok=True)
    var names = camera_names()
    for k in range(N_CAMS):
        var hwc = List[UInt8](length=CAM_W * CAM_H * 3, fill=UInt8(0))
        for c in range(3):
            for p in range(CAM_W * CAM_H):
                hwc[p * 3 + c] = frames[k][c * CAM_W * CAM_H + p]
        save_png(dir + "/" + names[k] + "_undistorted" + tag + ".png", hwc, CAM_W, CAM_H, 3)
        comptime S = 256 // OBS_PX
        comptime W = OBS_PX * S
        var img = List[UInt8](length=W * W * 3, fill=UInt8(0))
        for yy in range(W):
            for xx in range(W):
                for c in range(3):
                    var v = Float64(x[(3 * k + c) * PLANE + (yy // S) * OBS_PX + xx // S]) + 0.5
                    var b = Int(v * 255.0 + 0.5)
                    img[(yy * W + xx) * 3 + c] = UInt8(max(0, min(255, b)))
        save_png(dir + "/" + names[k] + "_policy_view" + tag + ".png", img, W, W, 3)
    # ⚠ THE POSE THE FRAMES WERE TAKEN AT, in model radians through the joint
    # map — what `tools/so101/sim_view_at_pose.mojo` renders the sim cameras
    # at, so a real frame and a sim frame of the SAME arm pose can be laid
    # side by side (a joint-map error shows as a different arm silhouette).
    var ps = String("")
    for j in range(len(q)):
        ps += ("" if j == 0 else " ") + String(q[j])
    with open(dir + "/pose" + tag + ".txt", "w") as f:
        f.write(ps + "\n")
    print("  snap: " + dir + "/{overhead,wrist}_{undistorted,policy_view}" + tag
          + ".png + pose" + tag + ".txt (" + ps + ")")


def main() raises:
    var args = List[String]()
    for a in argv():
        args.append(String(a))
    var project = _arg(args, "--project", "so101-tower")
    var role = _arg(args, "--role", "pixel_lift")
    var ckpt = _arg(args, "--ckpt", "")
    var devices_csv = _arg(args, "--devices", "/dev/soarm_cam_overhead,/dev/soarm_cam_wrist")
    var port_arg = _arg(args, "--port", "")
    var fourcc = _arg(args, "--fourcc", "")
    var undistort_dir = _arg(args, "--undistort", "projects/so101-tower/cameras")
    var arm_it = _flag(args, "--arm")
    var seconds = Int(_arg(args, "--seconds", "20"))
    var do_return = not _flag(args, "--no-return")
    var go_start = not _flag(args, "--no-start-pose")
    var snap_dir = _arg(args, "--snap", "")
    var grip_arg = _arg(args, "--gripper-sign", "")
    var step_ticks = Int(_arg(args, "--step-ticks", String(MAX_STEP_TICKS)))
    var force_dark = _flag(args, "--force-dark")
    var rec_dir = _arg(args, "--record", "")
    var sysid = _arg(args, "--sysid", "")
    var grip_off = Float64(_arg(args, "--grip-offset", "0"))

    print("=" * 74)
    print("PIXEL STUDENT on the physical SO-101 — sim-to-real")
    print("=" * 74)
    comptime if JOINT_VEL:
        print("  joint velocities: finite differences of the mapped joint"
              " angles over each tick (same map, same signs as the angles)")

    # ── the policy and its manifest ───────────────────────────────────────
    if ckpt.byte_length() == 0:
        ckpt = resolve_policy(project, role)
        if ckpt.byte_length() == 0:
            raise Error("pixel deploy: no policies/" + role + ".ckpt in project "
                        + project + " — promote one, or pass --ckpt")
        print("policy      " + describe_policy(project, role))
    print("weights     " + ckpt)
    var man_path = _manifest_beside(ckpt)
    print("manifest    " + man_path)
    var man = check_pixel_manifest(man_path)
    var grip_sign = man.gripper_sign
    if grip_arg.byte_length() > 0:
        grip_sign = grip_arg == "1"
    var period_ns = Int(man.control_period_s * 1e9)
    print("  trained on " + man.task + " | teacher " + man.teacher)
    print("  " + String(N_CAMS) + " cameras at " + String(OBS_PX) + "x"
          + String(OBS_PX) + " | control period " + fixed(man.control_period_s * 1000.0, 1)
          + " ms | gripper sign " + String(grip_sign))
    print("  action scale (rad per step at a = 1): arm " + String(man.delta_arm)
          + ", gripper " + String(man.delta_gripper) + " — the manifest's")
    var net = StudentNet.make["cpu", Kaiming](None)
    load_params["cpu"](net, ckpt, None)
    var x = Tensor.alloc(IN_DIM)
    var y = Tensor.alloc(ACT)
    var t0 = perf_counter_ns()
    for _i in range(5):
        net.forward["cpu", 1](TensorRefs[1](x), y, None)
    print("  forward (CPU, batch 1): " + fixed(Float64(perf_counter_ns() - t0) / 5e6, 2) + " ms")

    # ── the model: joint limits and the sim's start pose ──────────────────
    var f = load_family(String(FAMILY_PATH))
    var fmd = parse_model_runtime(scene_path(f))
    var jadr = List[Int]()
    var acc = 0
    for i in range(len(fmd.joints)):
        jadr.append(acc)
        acc += fmd.joints[i].nq
    var lo = Array[Float64, SO101_N](fill=0.0)
    var hi = Array[Float64, SO101_N](fill=0.0)
    var qa = List[Int]()
    print("")
    print("   slot  arm joint        model actuator            ctrlrange (rad)")
    for i in range(ACT):
        lo[i] = fmd.actuators[i].ctrl_min
        hi[i] = fmd.actuators[i].ctrl_max
        qa.append(jadr[fmd.actuators[i].joint_id])
        print("   " + String(i) + "     " + pad_right(joint_name(i), 16)
              + pad_right(fmd.actuator_names[i], 26) + "[" + col(lo[i], 7, 3)
              + "," + col(hi[i], 7, 3) + " ]")
    var q0 = posed_qpos[So101TowerPlacement](
        man.task, String(FAMILY), So101TowerConfig.SLOT_RADIUS
    )
    var q_start = List[Float64]()
    for i in range(ACT):
        q_start.append(q0[qa[i]])

    # ── the cameras ───────────────────────────────────────────────────────
    var devices = parse_camera_specs(devices_csv)
    if len(devices) != N_CAMS:
        raise Error("pixel deploy: " + String(len(devices)) + " devices for a "
                    + String(N_CAMS) + "-camera policy (overhead first)")
    var names = camera_names()
    var cams = List[CameraReader]()
    print("")
    for i in range(N_CAMS):
        print("camera slot " + String(i) + " = " + pad_right(names[i], 10)
              + " <- " + devices[i])
        var c = CameraReader.from_spec(
            devices[i], CAM_W, CAM_H, 30.0, rgb=False, fourcc=fourcc,
            out_w=CAM_W, out_h=CAM_H,
        )
        if undistort_dir == "none":
            print("   ⚠⚠ --undistort none: the policy will see FISHEYE frames it"
                  " was never trained on")
        else:
            var cp = undistort_dir + "/camera_" + names[i] + ".txt"
            c.set_undistort(cp, CAM_FOVY_DEG)
            print("            undistorted to the sim pinhole via " + cp)
        c.start(wait_ms=8000)
        print("            " + c.resolved_node() + "  " + c.negotiated_fourcc()
              + "  " + fixed(c.negotiated_fps(), 1) + " fps")
        cams.append(c^)
    var frames = List[List[UInt8]]()
    for i in range(N_CAMS):
        frames.append(List[UInt8](length=cams[i].frame_bytes(), fill=UInt8(0)))
        if not cams[i].take_blocking(frames[i], timeout_ms=4000):
            raise Error("pixel deploy: no first frame from " + devices[i])
    _warm_cameras(cams, frames, CAMERA_WARMUP_S)
    var xs = List[Scalar[DT]](length=IN_DIM, fill=Scalar[DT](0))
    for i in range(N_CAMS):
        frame_to_planes(frames[i], CAM_W, CAM_H, i, xs)
    # ⚠ THE BRIGHTNESS CHECK, against the sim's own policy view
    var too_dark = False
    for i in range(N_CAMS):
        var m = _view_mean(xs, i)
        var ref_m = SIM_MEAN_WRIST if names[i] == "wrist" else SIM_MEAN_OVERHEAD
        var mr = 0.0
        var mg = 0.0
        var mb = 0.0
        for p in range(PLANE):
            mr += Float64(xs[(3 * i) * PLANE + p]) + 0.5
            mg += Float64(xs[(3 * i + 1) * PLANE + p]) + 0.5
            mb += Float64(xs[(3 * i + 2) * PLANE + p]) + 0.5
        print("  " + pad_right(names[i], 9) + " policy view mean " + fixed(m, 3)
              + " (R " + fixed(mr / Float64(PLANE), 3) + " G "
              + fixed(mg / Float64(PLANE), 3) + " B " + fixed(mb / Float64(PLANE), 3)
              + ") | sim " + fixed(ref_m, 3)
              + ("   ⚠⚠ UNDER HALF THE SIM'S" if m < 0.5 * ref_m else ""))
        if m < 0.5 * ref_m:
            too_dark = True

    # ── the arm ───────────────────────────────────────────────────────────
    print("")
    var f_port = follower_port(port_arg)
    print("follower    " + f_port)
    var why = port_refusal(f_port, String("follower"))
    if why.byte_length() > 0:
        raise Error("pixel deploy: " + why)
    var arm = SO101Arm(f_port, max_step_ticks=step_ticks,
                       track_step_ticks=TRACK_STEP_TICKS)
    arm.bus.timeout_ms = 20
    var jmap = SimJointMap.tower_follower(arm.cal, lo.copy(), hi.copy())
    # the map must round-trip at interior points, or the pose is mirrored
    var worst = 0.0
    for i in range(SO101_N):
        for k in range(3):
            var v = lo[i] + (0.25 + 0.25 * Float64(k)) * (hi[i] - lo[i])
            var e = abs(jmap.to_sim(arm.cal, i, jmap.from_sim(arm.cal, i, v)) - v)
            if e > worst:
                worst = e
    if worst > 0.02:
        raise Error("pixel deploy: the joint map does not round-trip (worst "
                    + fixed(worst, 4) + " rad) — NOT arming")
    print("  joint map round-trips, worst " + fixed(worst, 4) + " rad")
    var raw = Array[Int32, SO101_N](fill=0)
    if arm.read_positions(Span(raw)) != SO101_N:
        raise Error("pixel deploy: the follower did not report 6 positions")
    var q = List[Float64](length=SO101_N, fill=0.0)
    print("")
    print("   joint          present (rad)   sim start (rad)")
    for i in range(SO101_N):
        q[i] = jmap.to_sim_unclamped(arm.cal, i, raw[i])
        var d = abs(q[i] - q_start[i])
        print("   " + pad_right(joint_name(i), 14) + col(q[i], 10, 3)
              + col(q_start[i], 16, 3)
              + ("   ⚠ far from the sim start" if d > 0.3 else ""))

    # one dry forward on the real observation, printed (the arm at rest:
    # velocities zero)
    if grip_off != 0.0:
        print("  --grip-offset", grip_off, "rad on the gripper angle the policy sees")
    var q_pol = q.copy()
    q_pol[SO101_N - 1] += grip_off
    joints_to_planes(q_pol, xs)
    var qd = List[Float64](length=SO101_N, fill=0.0)
    joint_vels_to_planes(qd, xs)
    # the last executed actions (TASK_PPO_ACT_HIST builds): none yet
    var hist = List[Float64](length=HIST_WORDS, fill=0.0)
    act_hist_to_planes(hist, xs)
    for k in range(IN_DIM):
        x.data[k] = xs[k]
    net.forward["cpu", 1](TensorRefs[1](x), y, None)
    var line = String("  first action on the real observation:")
    for j in range(ACT):
        line += " " + col(Float64(y.data[j]), 6, 2)
    print(line)
    if snap_dir.byte_length() > 0:
        _snap(snap_dir, frames, xs, q)

    if too_dark and arm_it and not force_dark:
        for i in range(N_CAMS):
            try:
                cams[i].stop()
            except:
                pass
        raise Error(
            "pixel deploy: a camera's picture is under half the sim's"
            " brightness — NOT arming (check the lights and the camera's"
            " exposure; --force-dark overrides)"
        )

    # ── go ────────────────────────────────────────────────────────────────
    var stdin = StdinReader()
    var interactive = stdin_is_tty()
    print("")
    if arm_it:
        print("⚠⚠ THE FOLLOWER WILL BE ENERGISED AND WILL MOVE FOR "
              + String(seconds) + " s"
              + (" (after ramping to the sim's start pose)." if go_start else "."))
    else:
        print("dry run — torque stays OFF and the arm is backdrivable.")
    print("press Enter to start (q = quit)"
          + (", and Enter again to stop early" if interactive else ""))
    stdin.discard_pending()
    var answer = stdin.line()
    if answer == "q" or answer == "Q":
        for i in range(N_CAMS):
            try:
                cams[i].stop()
            except:
                pass
        print("nothing was armed.")
        return

    var start_pose = List[Int32](length=SO101_N, fill=0)
    for i in range(SO101_N):
        start_pose[i] = raw[i]
    if arm_it:
        arm.set_position_mode()
        var hold = arm.max_step_ticks
        arm.max_step_ticks = 0
        arm.write_goals(Span(raw))
        arm.max_step_ticks = hold
        arm.set_torque(True)
        print("follower torque ON")
        if go_start:
            var tgt = List[Int32](length=SO101_N, fill=0)
            for i in range(SO101_N):
                tgt[i] = jmap.from_sim(arm.cal, i, q_start[i])
            print("ramping to the sim's start pose (<= "
                  + String(START_POSE_TIMEOUT_S) + " s) ...")
            if not _ramp_to(arm, tgt, START_POSE_TIMEOUT_S):
                print("⚠ did not reach the start pose — returning and stopping")
                _ = return_and_release(arm, start_pose, arm_it, do_return,
                                       stdin, interactive)
                for i in range(N_CAMS):
                    try:
                        cams[i].stop()
                    except:
                        pass
                return
            print("at the sim's start pose\n")
    else:
        print("dry run — nothing energised\n")

    # ── --sysid: the servos' step response, no policy ───────────────────
    if sysid.byte_length() > 0:
        if not arm_it:
            raise Error("pixel deploy: --sysid moves the arm; it needs --arm")
        var base_q = List[Float64](length=SO101_N, fill=0.0)
        if arm.read_positions(Span(raw)) == SO101_N:
            for i in range(SO101_N):
                base_q[i] = jmap.to_sim_unclamped(arm.cal, i, raw[i])
        var csv = String("t_s,joint,tgt,q0,q1,q2,q3,q4,q5\n")
        var sgoals = Array[Int32, SO101_N](fill=0)
        var t0s = perf_counter_ns()
        var hold = Int(0.8 / man.control_period_s)
        try:
            for jj in range(SO101_N):
                var amp = 0.3 if jj == SO101_N - 1 else 0.1
                var seq: List[Float64] = [0.0, amp, 0.0, -amp, 0.0]
                print("  sysid: joint " + joint_name(jj) + " +-" + fixed(amp, 2) + " rad")
                for k in range(len(seq)):
                    for _h in range(hold):
                        var tt = perf_counter_ns()
                        var tgt_j = base_q[jj] + seq[k]
                        if tgt_j < lo[jj]:
                            tgt_j = lo[jj]
                        if tgt_j > hi[jj]:
                            tgt_j = hi[jj]
                        for i in range(SO101_N):
                            var ti = tgt_j if i == jj else base_q[i]
                            sgoals[i] = jmap.from_sim(arm.cal, i, ti)
                        arm.write_goals(Span(sgoals))
                        if arm.read_positions(Span(raw)) == SO101_N:
                            var row = String(Float64(perf_counter_ns() - t0s) / 1e9) + "," + String(jj) + "," + String(tgt_j)
                            for i in range(SO101_N):
                                row += "," + String(jmap.to_sim_unclamped(arm.cal, i, raw[i]))
                            csv += row + "\n"
                        _spin_until(tt + period_ns)
        finally:
            with open(sysid, "w") as f:
                f.write(csv)
            print("  wrote " + sysid)
            var released = return_and_release(
                arm, start_pose, arm_it, do_return, stdin, interactive
            )
            if not released:
                print("⚠ the follower is STILL ENERGISED — deliberate, see above.")
            for i in range(N_CAMS):
                try:
                    cams[i].stop()
                except:
                    pass
        return

    var goals = Array[Int32, SO101_N](fill=0)
    var ticks = 0
    var stale = 0
    var bus_skipped = 0
    var clamped = 0
    var sum_fwd = 0.0
    var worst_tick = 0.0
    var rec_csv = String("t_s,q0,q1,q2,q3,q4,q5,qd0,qd1,qd2,qd3,qd4,qd5,a0,a1,a2,a3,a4,a5,tgt0,tgt1,tgt2,tgt3,tgt4,tgt5\n")
    if rec_dir.byte_length() > 0:
        makedirs(rec_dir, exist_ok=True)
        print("  recording to " + rec_dir + " (ticks.csv, frames every 16 ticks)")
    var loop_ns = 0
    # ⚠ THE JOINT VELOCITIES ARE FINITE DIFFERENCES of the mapped angles over
    # the measured tick, not the servos' own speed register: they go through
    # the SAME joint map as the angles, so their signs and units cannot
    # disagree with the positions the policy was trained beside. Tick
    # quantisation (~0.0015 rad / 32 ms ~ 0.05 rad/s) is 0.01 after the
    # plane's x0.2 — below the sim's own step noise.
    # ⚠ RE-READ HERE, AFTER THE RAMP: `q` still holds the pre-arm pose, and
    # the ramp to the sim's start moved the arm (0.41 rad of wrist flex on the
    # first bring-up) — differencing against it fed the first tick a 13 rad/s
    # "velocity".
    if arm.read_positions(Span(raw)) == SO101_N:
        for i in range(SO101_N):
            q[i] = jmap.to_sim_unclamped(arm.cal, i, raw[i])
    var q_prev = List[Float64](length=SO101_N, fill=0.0)
    for i in range(SO101_N):
        q_prev[i] = q[i]
    var t_prev = perf_counter_ns()
    var loop_t0 = perf_counter_ns()
    var deadline = loop_t0 + seconds * 1_000_000_000
    var a_ex = List[Float64](length=ACT, fill=0.0)
    var line2 = String("")
    var ra = String("")
    var rt = String("")
    if man.repeat > 1:
        print("  the policy acts every", man.repeat, "ticks (",
              fixed(1.0 / (man.control_period_s * Float64(man.repeat)), 1), "Hz)")
    try:
        while perf_counter_ns() < deadline:
            var tt = perf_counter_ns()
            if interactive and stdin.has_input():
                _ = stdin.line()
                print("  stopped by the operator")
                break
            # observe: the latest frame of each camera (the previous one if
            # none arrived this period — counted)
            for i in range(N_CAMS):
                if cams[i].take_latest(frames[i]) == 0:
                    stale += 1
                frame_to_planes(frames[i], CAM_W, CAM_H, i, xs)
            if arm.read_positions(Span(raw)) != SO101_N:
                bus_skipped += 1
                _spin_until(tt + period_ns)
                continue
            var t_now = perf_counter_ns()
            var dt_s = Float64(t_now - t_prev) / 1e9
            for i in range(SO101_N):
                q[i] = jmap.to_sim_unclamped(arm.cal, i, raw[i])
                qd[i] = (q[i] - q_prev[i]) / dt_s if dt_s > 1e-4 else 0.0
                q_prev[i] = q[i]
            t_prev = t_now
            for i in range(SO101_N):
                q_pol[i] = q[i]
            q_pol[SO101_N - 1] += grip_off
            joints_to_planes(q_pol, xs)
            joint_vels_to_planes(qd, xs)
            act_hist_to_planes(hist, xs)
            if snap_dir.byte_length() > 0 and ticks == 62:
                _snap(snap_dir, frames, xs, q, String("_t2s"))
            # the policy acts every `repeat` ticks (the manifest's cadence);
            # between, the last goals are re-sent and the joints still read
            # every tick (the velocities are per tick, as the sim's qvel)
            var acting = ticks % man.repeat == 0
            if acting:
                for k in range(IN_DIM):
                    x.data[k] = xs[k]
                var tf = perf_counter_ns()
                net.forward["cpu", 1](TensorRefs[1](x), y, None)
                sum_fwd += Float64(perf_counter_ns() - tf) / 1e6
                # act: the teacher's delta rule, then servo ticks
                line2 = String("")
                ra = String("")
                rt = String("")
                for j in range(ACT):
                    var a = Float64(student_act(y.data[j], j, grip_sign))
                    a_ex[j] = a
                    var tgt = delta_target(q[j], a, j, lo[j], hi[j], man.delta_arm, man.delta_gripper)
                    if tgt <= lo[j] or tgt >= hi[j]:
                        clamped += 1
                    goals[j] = jmap.from_sim(arm.cal, j, tgt)
                    line2 += " " + col(a, 6, 2)
                    ra += "," + String(a)
                    rt += "," + String(tgt)
            if rec_dir.byte_length() > 0:
                var row = String(Float64(perf_counter_ns() - loop_t0) / 1e9)
                for j in range(ACT):
                    row += "," + String(q[j])
                for j in range(ACT):
                    row += "," + String(qd[j])
                rec_csv += row + ra + rt + "\n"
                if ticks % 16 == 0:
                    _snap(rec_dir, frames, xs, q, String("_") + String(ticks))
            if arm_it:
                arm.write_goals(Span(goals))
            # ⚠ what was SENT, as the trainer records it (the dry run too: its
            # policy then sees the commands it would have made)
            if acting:
                act_hist_push(hist, a_ex)
            if ticks % 15 == 0:
                print("  t=" + pad_left(fixed(Float64(perf_counter_ns() - loop_t0) / 1e9, 1), 5)
                      + "s  a:" + line2)
            ticks += 1
            var dt = Float64(perf_counter_ns() - tt) / 1e6
            if dt > worst_tick:
                worst_tick = dt
            _spin_until(tt + period_ns)
    finally:
        loop_ns = perf_counter_ns() - loop_t0
        if rec_dir.byte_length() > 0:
            try:
                with open(rec_dir + "/ticks.csv", "w") as f:
                    f.write(rec_csv)
                print("  wrote " + rec_dir + "/ticks.csv")
            except:
                print("  ⚠ could not write the tick log")
        var released = return_and_release(
            arm, start_pose, arm_it, do_return, stdin, interactive
        )
        if not released:
            print("⚠ the follower is STILL ENERGISED — deliberate, see above.")
        for i in range(N_CAMS):
            try:
                cams[i].stop()
            except:
                pass

    var el = Float64(loop_ns) / 1e9
    print("=" * 74)
    print("pixel student run" + (" (ARMED)" if arm_it else " (dry)"))
    print("  ticks           = " + String(ticks) + " in " + fixed(el, 1) + " s = "
          + fixed(Float64(ticks) / max(el, 1e-9), 1) + " Hz (sim: "
          + fixed(1.0 / man.control_period_s, 2) + ")")
    print("  worst tick      = " + fixed(worst_tick, 1) + " ms | forward mean "
          + fixed(sum_fwd / Float64(max(ticks, 1)), 2) + " ms")
    print("  stale frames    = " + String(stale) + " | bus reads skipped "
          + String(bus_skipped) + " | targets at a joint limit " + String(clamped))
    print("=" * 74)
