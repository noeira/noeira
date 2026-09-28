# +--------------------------------------------------------------------------+ #
# | Where the camera is, in the robot's frame — solved with the arm as the ruler
# +--------------------------------------------------------------------------+ #
"""Capture arm-pose / marker-pose pairs and fit the camera -> base transform.

    pixi run build-opencv      # ONCE
    pixi run build-imgui       # ONCE
    pixi run build-serial      # ONCE

    pixi run mojo run -I . examples/so101/calibrate_camera_extrinsics.mojo \\
        --camera front --device 0 --marker-mm 30 \\
        --offset 0.0 -0.045 0.012

`docs/VISION_ASSESSMENT_2026_09_09.md` §2 item 1, the RIG half.
`noeira/vision/extrinsics.mojo` is the solver; this is what feeds it.

## ⚠⚠ THIS PROGRAM NEVER ENERGISES THE ARM, AND HAS NO FLAG THAT DOES

Every other program in this directory has an `--arm`. This one does not, and
that is a design decision rather than an omission: the calibration wants the
gripper at a DOZEN well-spread poses including awkward ones near the workspace
edges, and the fastest, safest and most accurate way to get there is to release
the torque and move it **by hand**. It releases torque at start-up and again in
a `finally`.

⚠ A `finally` DOES NOT COVER AN ABORT OR A SIGNAL. If this dies hard, run
`pixi run soarm-torque-off`.

## What it is doing

A marker taped to the gripper is seen by two instruments at once:

| | says where the marker is | in which frame |
|---|---|---|
| the camera | `solve_pnp` on the detected corners | CAMERA |
| the arm | FK on the measured `qpos`, plus `--offset` | ROBOT BASE |

N such pairs determine the rigid transform between the frames — Kabsch, in
`fit_rigid`. ⚠ THIS IS NOT HAND-EYE CALIBRATION AND DOES NOT NEED TO BE:
`cv2.calibrateHandEye` is not in this OpenCV 5.0 build, and a FIXED camera
admits the far easier 3D-3D problem.

## ⚠⚠ THE FOUR THINGS THAT WILL SILENTLY BE WRONG

1. **`--offset` IS NOT OPTIONAL, IT IS A MEASUREMENT.** It is the marker
   CENTRE in the gripper body's own frame, in metres. Leave it at zero and you
   are asserting the marker's centre coincides with the body origin — which is
   inside the plastic. The offset ROTATES with the wrist, so it is not a
   constant bias the fit can absorb: get it wrong by 10 mm and the residual
   rises by roughly 10 mm and stays there however many poses you take.

2. **THE MARKER GOES ON THE GRIPPER, NOT THE JAW.** `--body` defaults to
   `GRIPPER_BODY_IDX`, deliberately NOT the config's `EE_BODY` — which is
   `moving_jaw`, whose pose changes when the gripper opens. A marker on the
   jaw silently encodes the jaw angle into the fit.

3. **THE TWO READINGS MUST BE SIMULTANEOUS.** A frame taken while the arm is
   moving pairs a pose with a marker that was somewhere else. Capture is
   therefore REFUSED unless the arm has been still for `STILL_TICKS` ticks;
   the panel says so.

4. **SPREAD, NOT COUNT.** Twenty poses in one corner of the workspace fit
   beautifully and determine almost nothing. The panel reports the three
   principal extents of what has been captured, and the third one is the one
   to watch: sweeping the gripper across a table gives a plane, and a brick
   picked 15 cm above that plane is then localised by a direction nobody
   measured.

## Every capture is written to `<calib>.poses.txt`

One line per pose, rewritten at each capture/drop/clear: the marker in the
camera frame, the gripper's FK position and rotation, the six joint values
(model radians) and raw servo ticks, and the marker's four corner pixels.
So a residual that will not come down can be diagnosed OFFLINE — a scale
(marker size, focal length), a joint zero offset, distance-dependent depth
noise, the lens edge — instead of by recapturing. The panel's `scale` line
is the first of those: the least-squares scale between the camera-side and
arm-side point sets at the fitted rotation. It should read 1.000; a
systematic scale error grows the residual with the spread.

## Without `--offset`, the offset is SOLVED (`fit_rigid_with_offset`)

Leave `--offset` out and the marker's position on the gripper is estimated
together with the camera, from the captured poses themselves — the marker can
then go anywhere rigid on the gripper body (the rig's wrist-camera mount is
a good flat spot) without measuring anything but its printed size. It needs
the WRIST TURNED between captures, roll AND pitch: with one orientation the
offset is undetermined, and the fit says so instead of guessing (`wrist`
spread below 10 deg is refused). 10+ poses.

## The arm's joint zero (`--joint-zero follower|none`)

The fit trusts the arm's FK, so it trusts the servo -> model joint map. The
so101-tower follower's calibrated zero is NOT the model's (pan -10.7 deg,
`robot/so101/sim_map.mojo`), and with the reference map (`none`) the fit
absorbs that into a camera 12 deg off and a 12 mm residual — measured, on
this rig, 2026-09-22. `follower` (the default) applies the measured zero and
REFUSES an arm whose calibration is not the one it was measured against.

## Fisheye cameras (the so101-tower rig)

A `model fisheye` calibration (`examples/vision/calibrate_fisheye.mojo`) is
handled by UNDISTORTING the four marker corners through the lens
(`FisheyeLens.unproject`) and solving the pinhole problem on the normalised
coordinates (`K = I`, no distortion). Detection runs on a 2x upscale by
default for a fisheye (`--detect-scale`), for the reason that tool gives:
at 640x480 the lens makes a marker a few pixels wide.

`--sim-camera NAME` (default `<camera>_cam`, e.g. `overhead_cam`) compares
the fitted pose with the SIMULATOR's camera of that name in the
`so101_tower` scene. The robot base sits at the family's `base_pos` in that
scene, so the fit is moved there first. The tool prints the position and
orientation error and the corrected `<camera pos=... xyaxes=...>`, in the
camera's PARENT BODY frame, ready to paste into the asset. The pose is
printed, not written: the scene is shared, and changing it changes every
render.

## What it needs first

An INTRINSICS calibration for this camera, as a
`mojo-rl-camera-calibration` file — `examples/vision/camera_studio.mojo`
produces one. This program refuses to start without it, because `solve_pnp`
with a guessed focal length returns a pose with an unknown scale factor on it,
and a scale error in the correspondences becomes a rotation error in the fit.
"""

from std.math import acos
from std.sys import argv
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from noeira.math3d import Mat3 as Mat3Generic, Quat as QuatGeneric, Vec3 as Vec3Generic
from noeira.nn.constants import DT
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.envs.robots.so_arm101_xml import SoArm101Model, GRIPPER_BODY_IDX
from noeira.envs.robots.so_arm101 import SoArm101ReachConfig
from noeira.physics3d.fields import actuator_column
from noeira.physics3d.gpu.constants import ACT_IDX_CTRL_MAX, ACT_IDX_CTRL_MIN
from noeira.render.imgui import (
    IgTexture,
    ig_begin_panel,
    ig_begin_window,
    ig_button,
    ig_end,
    ig_framerate,
    ig_last_item_rect,
    ig_overlay_line,
    ig_same_line,
    ig_separator,
    ig_separator_text,
    ig_text,
    ig_text_colored,
    ig_text_disabled,
    imgui_shim_available,
)
from noeira.render.renderer3d import Renderer3D
from noeira.robot.so101 import SO101Arm, SO101_N, joint_short
from noeira.robot.so101.ports import follower_port
from noeira.robot.so101.sim_map import SimJointMap
from noeira.utils.fmt import fixed
from noeira.vision.calib_file import CameraCalib, read_calib, write_calib
from noeira.vision.extrinsics import RigidFit, fit_rigid, fit_rigid_with_offset
from noeira.vision.camera_thread import open_camera_spec
from noeira.vision.fisheye import FisheyeLens
from noeira.vision.preprocess import pil_bilinear_u8
from noeira.tasks.so101_tower_camera_pose import (
    SimCamera, tower_sim_camera, camera_pose_vs_sim, fit_to_mujoco_rot,
)
from noeira.vision.opencv import (
    ArucoDetector,
    DICT_4X4_50,
    SOLVEPNP_IPPE_SQUARE,
    VideoCapture,
    opencv_shim_available,
    solve_pnp,
)

comptime Vec3d = Vec3Generic[DType.float64]
comptime Mat3d = Mat3Generic[DType.float64]
comptime Quatd = QuatGeneric[DType.float64]


comptime WIN_W = 1220
comptime WIN_H = 780
comptime REQ_W = 640
comptime REQ_H = 480

comptime EnvT = Phyics3dEnv[
    SoArm101Model, SoArm101ReachConfig, DT, TERMINATE_ON_UNHEALTHY=False
]

comptime STILL_EPS_TICKS = 4
"""Raw servo ticks. Below this per-joint change the arm counts as still.
⚠ NOT ZERO: a released servo reports 1-2 ticks of jitter forever, so a
zero threshold means the capture button is never enabled."""

comptime STILL_TICKS = 6
"""Consecutive still ticks required. At the loop's rate this is a fraction of
a second — long enough that a hand let go of the arm, short enough not to be
annoying."""

comptime MIN_POSES = 6
"""⚠ THE SOLVER ACCEPTS 3 AND THREE PROVES NOTHING — it fits exactly, so the
residual is 0 whatever the data says. This is the number at which `rms_mm`
starts being a measurement rather than an identity."""
comptime MIN_POSES_AUTO = 10
"""With the offset SOLVED (no `--offset`): three more unknowns, so more poses
before the residual means as much."""


def _refit(
    auto_off: Bool, ref cam_pts: List[Float64], ref grip_pos: List[Float64],
    ref grip_rot: List[Float64], off: Vec3d, mut solved_off: Vec3d,
    mut wrist_spread: Float64,
) raises -> RigidFit:
    """The fit over the captured poses: the offset SOLVED (`auto_off`) or
    the given one applied."""
    var n = len(cam_pts) // 3
    if auto_off:
        if n < 5:
            raise String("offset solve: capture 5+ poses (turn the wrist between them)")
        var of = fit_rigid_with_offset(cam_pts, grip_pos, grip_rot)
        solved_off = of.offset
        wrist_spread = of.rot_spread_deg
        return of.fit.copy()
    var base = List[Float64]()
    for k in range(n):
        var rk = Mat3d(
            grip_rot[k * 9], grip_rot[k * 9 + 1], grip_rot[k * 9 + 2],
            grip_rot[k * 9 + 3], grip_rot[k * 9 + 4], grip_rot[k * 9 + 5],
            grip_rot[k * 9 + 6], grip_rot[k * 9 + 7], grip_rot[k * 9 + 8],
        )
        var b = Vec3d(grip_pos[k * 3], grip_pos[k * 3 + 1], grip_pos[k * 3 + 2]) + rk * off
        base.append(b.x)
        base.append(b.y)
        base.append(b.z)
    solved_off = off
    return fit_rigid(cam_pts, base)


def _dump_poses(
    path: String, ref cam: List[Float64], ref gp: List[Float64],
    ref gr: List[Float64], ref q: List[Float64], ref raw: List[Int],
    ref px: List[Float64],
):
    """`<calib>.poses.txt` — every capture, rewritten whole (see the header).
    A failure to write is printed, never raised: the capture itself stands."""
    var s = String(
        "# extrinsics captures: cam_x cam_y cam_z (m, camera frame) | grip_x"
        " grip_y grip_z (m, base) | grip_rot r00..r22 (row-major) | q0..q5"
        " (model rad) | raw0..raw5 (ticks) | u0 v0 .. u3 v3 (marker corners, px)\n"
    )
    var n = len(cam) // 3
    for k in range(n):
        for c in range(3):
            s += String(cam[k * 3 + c]) + " "
        s += "| "
        for c in range(3):
            s += String(gp[k * 3 + c]) + " "
        s += "| "
        for c in range(9):
            s += String(gr[k * 9 + c]) + " "
        s += "| "
        for c in range(SO101_N):
            s += String(q[k * SO101_N + c]) + " "
        s += "| "
        for c in range(SO101_N):
            s += String(raw[k * SO101_N + c]) + " "
        s += "| "
        for c in range(8):
            s += String(px[k * 8 + c]) + " "
        s += "\n"
    try:
        with open(path, "w") as f:
            f.write(s)
    except e:
        print("could not write", path, "-", e)


def _fit_scale(
    fit: RigidFit, ref cam: List[Float64], ref gp: List[Float64],
    ref gr: List[Float64], off: Vec3d,
) -> Float64:
    """Least-squares scale between the camera-side points (rotated by the
    fit) and the arm-side points, both centred: 1.0 when the marker size and
    the focal length are right."""
    var n = len(cam) // 3
    if n < 2:
        return 1.0
    var cc = Vec3d.zero()
    var cb = Vec3d.zero()
    var pc = List[Vec3d]()
    var pb = List[Vec3d]()
    for k in range(n):
        var rk = Mat3d(
            gr[k * 9], gr[k * 9 + 1], gr[k * 9 + 2], gr[k * 9 + 3], gr[k * 9 + 4],
            gr[k * 9 + 5], gr[k * 9 + 6], gr[k * 9 + 7], gr[k * 9 + 8],
        )
        var c = fit.rot * Vec3d(cam[k * 3], cam[k * 3 + 1], cam[k * 3 + 2])
        var b = Vec3d(gp[k * 3], gp[k * 3 + 1], gp[k * 3 + 2]) + rk * off
        pc.append(c)
        pb.append(b)
        cc = cc + c
        cb = cb + b
    cc = cc / Float64(n)
    cb = cb / Float64(n)
    var num = 0.0
    var den = 0.0
    for k in range(n):
        var dc = pc[k] - cc
        num += Float64(dc.dot(pb[k] - cb))
        den += Float64(dc.dot(dc))
    return num / den if den > 0.0 else 1.0


def _fmt3(v: Vec3d, scale: Float64, digits: Int) -> String:
    return (
        fixed(v.x * scale, digits)
        + " "
        + fixed(v.y * scale, digits)
        + " "
        + fixed(v.z * scale, digits)
    )


def main() raises:
    # ── arguments ───────────────────────────────────────────────────────────
    var cam_name = String("front")
    var device_spec = String("0")
    var calib_path = String("")
    var marker_mm = 30.0
    var marker_id = -1
    var body = GRIPPER_BODY_IDX
    var off = Vec3d.zero()
    var port = follower_port()
    var detect_scale = -1
    var sim_cam_name = String("")
    var joint_zero = String("follower")
    var args = argv()
    for i in range(1, len(args)):
        var a = String(args[i])
        if a == "--camera" and i + 1 < len(args):
            cam_name = String(args[i + 1])
        elif a == "--device" and i + 1 < len(args):
            device_spec = String(args[i + 1])
        elif a == "--calib" and i + 1 < len(args):
            calib_path = String(args[i + 1])
        elif a == "--marker-mm" and i + 1 < len(args):
            marker_mm = Float64(String(args[i + 1]))
        elif a == "--marker-id" and i + 1 < len(args):
            marker_id = Int(String(args[i + 1]))
        elif a == "--body" and i + 1 < len(args):
            body = Int(String(args[i + 1]))
        elif a == "--port" and i + 1 < len(args):
            port = String(args[i + 1])
        elif a == "--detect-scale" and i + 1 < len(args):
            detect_scale = Int(String(args[i + 1]))
        elif a == "--sim-camera" and i + 1 < len(args):
            sim_cam_name = String(args[i + 1])
        elif a == "--joint-zero" and i + 1 < len(args):
            joint_zero = String(args[i + 1])
        elif a == "--offset" and i + 3 < len(args):
            off = Vec3d(
                Float64(String(args[i + 1])),
                Float64(String(args[i + 2])),
                Float64(String(args[i + 3])),
            )
    if calib_path == "":
        calib_path = String("scratch/camera_") + cam_name + ".txt"

    if not opencv_shim_available():
        print("OpenCV shim not built.  Run:  pixi run build-opencv")
        return
    if not imgui_shim_available():
        print("Dear ImGui shim not built.  Run:  pixi run build-imgui")
        return

    # ── the intrinsics, which are a PREREQUISITE and not a nicety ──────────
    var calib: CameraCalib
    try:
        calib = read_calib(calib_path)
    except e:
        print("could not read the intrinsics for camera", cam_name, "-", e)
        print("  Expected a mojo-rl-camera-calibration file at:", calib_path)
        print("  Produce one with:  pixi run camera-studio --device", device_spec)
        print("  ⚠ WITHOUT IT solve_pnp has a guessed focal length, every")
        print("    marker distance carries that error, and a scale error in")
        print("    the correspondences becomes a ROTATION error in the fit.")
        return
    print("intrinsics:", calib_path)
    print("  fx", calib.fx, " fy", calib.fy, " cx", calib.cx, " cy", calib.cy)
    if calib.has_extrinsics:
        print(
            "  ⚠ this file ALREADY has extrinsics (rms",
            calib.rms_mm,
            "mm from",
            calib.poses,
            "poses) — saving will REPLACE them",
        )

    var auto_off = off.x == 0.0 and off.y == 0.0 and off.z == 0.0
    if auto_off:
        print("")
        print("no --offset: the marker's position on the gripper is SOLVED with")
        print("the camera. Turn the wrist (roll AND pitch) between captures;")
        print(String(MIN_POSES_AUTO) + "+ poses. The solved offset is printed.")
        print("")
    var min_poses = MIN_POSES_AUTO if auto_off else MIN_POSES

    # ── the camera ─────────────────────────────────────────────────────────
    var bgr = List[UInt8]()
    var cap = VideoCapture.closed()
    try:
        cap = open_camera_spec(device_spec, REQ_W, REQ_H, 30.0)
    except e:
        print("could not open camera", device_spec, "-", e)
        return
    if not cap.read(bgr):
        print("camera", device_spec, "opened but produced no frame")
        cap.close()
        return
    var fw = cap.width
    var fh = cap.height
    # ⚠ AGAINST WHAT THE DEVICE GAVE, never what was requested — OpenCV
    # substitutes a resolution and reports no error when it does.
    try:
        calib.require_size(fw, fh)
    except e:
        print(e)
        cap.close()
        return
    print("camera:", device_spec, "->", fw, "x", fh)

    # ── the arm, released ──────────────────────────────────────────────────
    print("opening", port, "...")
    var arm = SO101Arm(port, max_step_ticks=0)
    arm.bus.timeout_ms = 20
    arm.set_torque(False)
    print("  torque RELEASED — move the gripper by hand")

    # ── the kinematics oracle ──────────────────────────────────────────────
    var ctx = DeviceContext()
    var env = EnvT(ctx)
    _ = env.reset()
    var sf = SoArm101Model.make_spec_fields[DType.float64]()
    var lo_col = actuator_column(sf, ACT_IDX_CTRL_MIN, SO101_N)
    var hi_col = actuator_column(sf, ACT_IDX_CTRL_MAX, SO101_N)
    var lo = Array[Float64, SO101_N](fill=0.0)
    var hi = Array[Float64, SO101_N](fill=0.0)
    for i in range(SO101_N):
        lo[i] = Float64(lo_col[i])
        hi[i] = Float64(hi_col[i])
    var jmap: SimJointMap
    if joint_zero == "follower":
        jmap = SimJointMap.tower_follower(arm.cal, lo^, hi^)
    elif joint_zero == "none":
        jmap = SimJointMap.identity(lo^, hi^)
    else:
        raise Error("--joint-zero none|follower, not '" + joint_zero + "'")
    print("  " + jmap.describe())

    # ── the window ─────────────────────────────────────────────────────────
    var r = Renderer3D(WIN_W, WIN_H)
    var title = String("extrinsics — camera ") + cam_name
    r.init(title)
    if not r.imgui_init():
        print("ImGui declined this device")
        cap.close()
        return
    var tex = IgTexture(r.device.value(), fw, fh)
    var rgba = List[UInt8](unsafe_uninit_length=fw * fh * 4)
    var det = ArucoDetector(DICT_4X4_50)
    var ids = List[Int32]()
    var corners = List[Float32]()

    # ── state ──────────────────────────────────────────────────────────────
    var raw = Array[Int32, SO101_N](fill=0)
    var prev_raw = Array[Int32, SO101_N](fill=0)
    var qp = List[Float64](length=SO101_N, fill=0.0)
    var qv = List[Float64](length=SO101_N, fill=0.0)
    var still = 0

    var cam_pts = List[Float64]()
    # per pose: the gripper body's FK position (3) and rotation (9, row-major)
    # — the marker is at `grip_pos + grip_rot * offset`
    var grip_pos = List[Float64]()
    var grip_rot = List[Float64]()
    var solved_off = off
    # per pose, for `<calib>.poses.txt` (see the header)
    var pose_q = List[Float64]()
    var pose_raw = List[Int]()
    var pose_px = List[Float64]()
    var marker_px = List[Float64](length=8, fill=0.0)
    var poses_path = calib_path + ".poses.txt"
    var wrist_spread = 0.0
    var g_pos = Vec3d.zero()
    var g_rot = Mat3d.identity()
    var fit = RigidFit(
        Mat3d.identity(), Vec3d.zero(), 0.0, 0.0, 0, 0,
        Array[Float64, 3](fill=0.0),
    )
    var have_fit = False
    var fit_msg = String("")
    var status = String("release the arm and show the marker")

    # ⚠ `solve_pnp` takes OpenCV's radial-tangential vector; a fisheye file's
    # four Kannala-Brandt terms passed there would be a different lens that
    # still returns a pose. So a fisheye calibration undistorts the CORNERS
    # (`FisheyeLens.unproject`) and solves on normalised coordinates, K = I.
    var fisheye = calib.model == "fisheye"
    var lens = FisheyeLens(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, fw, fh)
    var k = calib.k_matrix()
    var dist = calib.dist.copy()
    if fisheye:
        lens = FisheyeLens.from_calib(calib)
        k = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        dist = List[Float64]()
    elif calib.model != "pinhole":
        raise Error("calibrate_camera_extrinsics: unknown model " + calib.model)
    if detect_scale < 1:
        detect_scale = 2 if fisheye else 1
    var up = List[UInt8]()
    if sim_cam_name == "":
        sim_cam_name = cam_name + "_cam"
    var sim = tower_sim_camera(sim_cam_name)
    if sim.found:
        print("sim camera:", sim.name, "at", _fmt3(sim.pos, 1000.0, 1), "mm (world)")
    else:
        print("sim camera: none named *" + sim_cam_name + " in the tower scene — no comparison")
    print("lens:", calib.model, "| detection at", detect_scale, "x")
    var obj = List[Float64]()
    var half = marker_mm / 2000.0
    obj.append(-half); obj.append(half); obj.append(0.0)
    obj.append(half); obj.append(half); obj.append(0.0)
    obj.append(half); obj.append(-half); obj.append(0.0)
    obj.append(-half); obj.append(-half); obj.append(0.0)
    var rvec = List[Float64]()
    var tvec = List[Float64]()


    try:
        while not r.check_quit():
            # ── the camera ──────────────────────────────────────────────────
            if not cap.read(bgr):
                # ⚠ PRINTED, NOT PUT IN `status`: the loop breaks here, so the
                # panel that would have shown it is never drawn again.
                print("the camera stopped delivering frames")
                break
            var n = fw * fh
            for i in range(n):
                rgba[i * 4 + 0] = bgr[i * 3 + 2]
                rgba[i * 4 + 1] = bgr[i * 3 + 1]
                rgba[i * 4 + 2] = bgr[i * 3 + 0]
                rgba[i * 4 + 3] = 255
            _ = tex.upload(rgba)
            var n_markers: Int
            if detect_scale > 1:
                pil_bilinear_u8(bgr, fw, fh, 3, up, fw * detect_scale, fh * detect_scale)
                n_markers = det.detect(up, fw * detect_scale, fh * detect_scale, 3, ids, corners)
                var sc = Float32(detect_scale)
                for q in range(n_markers * 8):
                    corners[q] = (corners[q] + 0.5) / sc - 0.5
            else:
                n_markers = det.detect(bgr, fw, fh, 3, ids, corners)

            # ── which marker is on the gripper ─────────────────────────────
            #
            # ⚠⚠ AMBIGUITY IS REFUSED, NOT GUESSED. Another marker in frame —
            # one taped to the table, or the calibration board still lying
            # there — would otherwise be paired with the arm's pose and drag
            # the whole fit toward it, with a residual that only says "these
            # points do not agree".
            var pick = -1
            var seen_ids = String("")
            for m in range(n_markers):
                if m > 0:
                    seen_ids += ","
                seen_ids += " " + String(ids[m])
                if marker_id >= 0:
                    if Int(ids[m]) == marker_id:
                        pick = m
                elif n_markers == 1:
                    pick = m

            var have_marker = False
            var p_cam = Vec3d.zero()
            if pick >= 0:
                var img_xy = List[Float64]()
                try:
                    for i in range(4):
                        var u = Float64(corners[pick * 8 + i * 2])
                        var v = Float64(corners[pick * 8 + i * 2 + 1])
                        if fisheye:
                            var ab = lens.unproject(u, v)
                            img_xy.append(ab[0])
                            img_xy.append(ab[1])
                        else:
                            img_xy.append(u)
                            img_xy.append(v)
                    solve_pnp(
                        obj, img_xy, k, dist, rvec, tvec, SOLVEPNP_IPPE_SQUARE
                    )
                    p_cam = Vec3d(tvec[0], tvec[1], tvec[2])
                    for q in range(8):
                        marker_px[q] = Float64(corners[pick * 8 + q])
                    have_marker = True
                except:
                    have_marker = False

            # ── the arm, and FK ────────────────────────────────────────────
            var have_arm = arm.read_positions(Span(raw)) == SO101_N
            var p_base = Vec3d.zero()
            if have_arm:
                var moved = 0
                for i in range(SO101_N):
                    var d = Int(raw[i]) - Int(prev_raw[i])
                    if d < 0:
                        d = -d
                    if d > moved:
                        moved = d
                    prev_raw[i] = raw[i]
                if moved <= STILL_EPS_TICKS:
                    still += 1
                else:
                    still = 0
                for i in range(SO101_N):
                    qp[i] = jmap.to_sim(arm.cal, i, raw[i])
                env.set_state(qp, qv)
                # ⚠⚠ `Data.xquat` IS PACKED (x, y, z, w) — W LAST, which is
                # NOT MuJoCo's (w, x, y, z), and `Quat` takes (w, x, y, z).
                # Taking the two for each other silently rotates `off` below
                # by whatever the wrist happens to be doing: no error, a
                # plausible point, and a fit that never converges below a
                # couple of centimetres.
                #
                # ⚠ THE AUTHORITY IS `forward_kinematics.mojo:134-137`, which
                # WRITES it in this order, and the identical conversion in
                # `tests/robots/test_so_arm101_camera_vs_mujoco.mojo:95-100`,
                # which is gated against `mjData.cam_xpos`. That makes this
                # the SECOND site of one rule — the shape this tree records
                # as its most recurring defect — so if a third appears, hoist
                # all three into `kinematics/`.
                var bq = Quatd(
                    Float64(env.d.xquat.data[body * 4 + 3]),
                    Float64(env.d.xquat.data[body * 4 + 0]),
                    Float64(env.d.xquat.data[body * 4 + 1]),
                    Float64(env.d.xquat.data[body * 4 + 2]),
                )
                var bp = Vec3d(
                    Float64(env.d.xpos.data[body * 3 + 0]),
                    Float64(env.d.xpos.data[body * 3 + 1]),
                    Float64(env.d.xpos.data[body * 3 + 2]),
                )
                g_pos = bp
                g_rot = Mat3d.from_quat(bq)
                p_base = bp + g_rot * solved_off
            else:
                still = 0

            var can_capture = have_marker and have_arm and still >= STILL_TICKS

            # ── UI ─────────────────────────────────────────────────────────
            r.imgui_new_frame()
            _ = ig_begin_panel(
                String("extrinsics"), 0.0, 0.0, 330.0, Float32(WIN_H)
            )
            ig_separator_text(String("camera"))
            ig_text(String("name    ") + cam_name)
            ig_text(
                String("size    ") + String(fw) + " x " + String(fh)
            )
            ig_text(String("fx      ") + fixed(calib.fx, 1))
            ig_text(String("ui      ") + fixed(Float64(ig_framerate()), 0) + " fps")

            ig_separator_text(String("marker"))
            if n_markers == 0:
                ig_text_disabled(String("none in frame"))
            else:
                ig_text(String("seen   ") + seen_ids)
            if pick < 0 and n_markers > 1 and marker_id < 0:
                ig_text_colored(
                    String("AMBIGUOUS — pass --marker-id"), 1.0, 0.5, 0.3, 1.0
                )
            elif have_marker:
                ig_text(String("cam    ") + _fmt3(p_cam, 1000.0, 1) + " mm")
            else:
                ig_text_disabled(String("cam    -"))

            ig_separator_text(String("arm"))
            if have_arm:
                var line = String("")
                for i in range(SO101_N):
                    if i > 0:
                        line += " "
                    line += joint_short(i)
                ig_text_disabled(line)
                var vals = String("")
                for i in range(SO101_N):
                    if i > 0:
                        vals += " "
                    vals += fixed(qp[i], 2)
                ig_text(vals)
                ig_text(String("base   ") + _fmt3(p_base, 1000.0, 1) + " mm")
            else:
                ig_text_colored(String("no reply from the bus"), 1.0, 0.4, 0.3, 1.0)
                ig_text_disabled(String("base   -"))
            if still >= STILL_TICKS:
                ig_text(String("still  yes"))
            else:
                ig_text_colored(
                    String("still  MOVING — hold it"), 1.0, 0.75, 0.2, 1.0
                )

            ig_separator_text(String("captures"))
            ig_text(String("poses  ") + String(len(cam_pts) // 3))
            if ig_button(String("capture pose"), 150.0, 30.0) and can_capture:
                cam_pts.append(p_cam.x)
                cam_pts.append(p_cam.y)
                cam_pts.append(p_cam.z)
                grip_pos.append(g_pos.x)
                grip_pos.append(g_pos.y)
                grip_pos.append(g_pos.z)
                for rr in range(3):
                    var row = g_rot.row(rr)
                    grip_rot.append(row.x)
                    grip_rot.append(row.y)
                    grip_rot.append(row.z)
                for jq in range(SO101_N):
                    pose_q.append(qp[jq])
                    pose_raw.append(Int(raw[jq]))
                for q in range(8):
                    pose_px.append(marker_px[q])
                _dump_poses(poses_path, cam_pts, grip_pos, grip_rot, pose_q, pose_raw, pose_px)
                try:
                    fit = _refit(auto_off, cam_pts, grip_pos, grip_rot, off, solved_off, wrist_spread)
                    have_fit = True
                    fit_msg = String("")
                except e:
                    have_fit = False
                    fit_msg = String(e)
                status = (
                    String("captured pose ") + String(len(cam_pts) // 3)
                )
            ig_same_line()
            if ig_button(String("drop worst"), 130.0, 30.0) and have_fit:
                # ⚠ ONE OUTLIER IS A MIS-TAKEN POSE, not noise: a marker
                # detected a frame late, or the arm nudged between the two
                # reads. Dropping it is right; dropping until the residual
                # looks nice is fitting the report.
                var w = fit.worst
                var keep_cam = List[Float64]()
                var keep_gp = List[Float64]()
                var keep_gr = List[Float64]()
                var keep_q = List[Float64]()
                var keep_raw = List[Int]()
                var keep_px = List[Float64]()
                for j in range(len(cam_pts) // 3):
                    if j == w:
                        continue
                    for c in range(3):
                        keep_cam.append(cam_pts[j * 3 + c])
                        keep_gp.append(grip_pos[j * 3 + c])
                    for c in range(9):
                        keep_gr.append(grip_rot[j * 9 + c])
                    for c in range(SO101_N):
                        keep_q.append(pose_q[j * SO101_N + c])
                        keep_raw.append(pose_raw[j * SO101_N + c])
                    for c in range(8):
                        keep_px.append(pose_px[j * 8 + c])
                cam_pts = keep_cam^
                grip_pos = keep_gp^
                grip_rot = keep_gr^
                pose_q = keep_q^
                pose_raw = keep_raw^
                pose_px = keep_px^
                _dump_poses(poses_path, cam_pts, grip_pos, grip_rot, pose_q, pose_raw, pose_px)
                have_fit = False
                try:
                    fit = _refit(auto_off, cam_pts, grip_pos, grip_rot, off, solved_off, wrist_spread)
                    have_fit = True
                    fit_msg = String("")
                except e:
                    fit_msg = String(e)
                status = String("dropped pose ") + String(w)
            if ig_button(String("clear")):
                cam_pts = List[Float64]()
                grip_pos = List[Float64]()
                grip_rot = List[Float64]()
                pose_q = List[Float64]()
                pose_raw = List[Int]()
                pose_px = List[Float64]()
                solved_off = off
                have_fit = False
                fit_msg = String("")
                status = String("cleared")
            if not can_capture:
                if not have_marker:
                    ig_text_disabled(String("capture needs the marker"))
                elif not have_arm:
                    ig_text_disabled(String("capture needs the arm"))
                else:
                    ig_text_disabled(String("capture needs the arm STILL"))
            else:
                ig_text(String("ready — move, hold, capture"))

            ig_separator_text(String("fit"))
            if have_fit:
                ig_text(String("rms    ") + fixed(fit.rms_mm, 2) + " mm")
                ig_text(
                    String("worst  ")
                    + fixed(fit.max_mm, 2)
                    + " mm at "
                    + String(fit.worst)
                )
                # ⚠⚠ THE THIRD EXTENT IS THE ONE THAT MATTERS. See the header.
                var sp = (
                    fixed(fit.spread_mm[0], 0)
                    + " / "
                    + fixed(fit.spread_mm[1], 0)
                    + " / "
                    + fixed(fit.spread_mm[2], 0)
                )
                if fit.spread_mm[2] < 30.0:
                    ig_text_colored(
                        String("spread ") + sp + " FLAT", 1.0, 0.75, 0.2, 1.0
                    )
                    ig_text_disabled(String("move the gripper UP and DOWN too"))
                else:
                    ig_text(String("spread ") + sp + " mm")
                    ig_text_disabled(String("(mm, three principal axes)"))
                ig_text(String("origin ") + _fmt3(fit.trans, 1.0, 3) + " m")
                ig_text(String("scale  ") + fixed(_fit_scale(fit, cam_pts, grip_pos, grip_rot, solved_off), 4)
                        + "  (1.000 = no scale error)")
                if auto_off:
                    ig_text(String("offset ") + _fmt3(solved_off, 1000.0, 1) + " mm (solved)")
                    ig_text(String("wrist  ") + fixed(wrist_spread, 0) + " deg spread")
                if sim.found:
                    var r_mj = fit_to_mujoco_rot(fit.rot)
                    var dpos = (fit.trans + sim.base_off - sim.pos) * 1000.0
                    var rel = sim.rot.transpose() @ r_mj
                    var cc = (Float64(rel.trace()) - 1.0) / 2.0
                    cc = 1.0 if cc > 1.0 else (-1.0 if cc < -1.0 else cc)
                    ig_text(String("vs sim ") + fixed(Float64(dpos.length()), 1) + " mm, "
                            + fixed(acos(cc) * 180.0 / 3.141592653589793, 2) + " deg")
            else:
                ig_text_disabled(String("rms    -"))
                ig_text_disabled(String("worst  -"))
                ig_text_disabled(String("spread -"))
                ig_text_disabled(String("origin -"))
            if fit_msg != "":
                ig_text_colored(fit_msg, 1.0, 0.5, 0.3, 1.0)

            var enough = have_fit and len(cam_pts) // 3 >= min_poses
            if ig_button(String("save calibration"), 200.0, 30.0) and enough:
                calib.has_extrinsics = True
                calib.rot = fit.rot
                calib.trans = fit.trans
                calib.rms_mm = fit.rms_mm
                calib.poses = fit.n
                try:
                    write_calib(calib_path, calib)
                    status = String("saved to ") + calib_path
                    print("saved", calib_path)
                    print(String(fit))
                    print("marker offset", _fmt3(solved_off, 1000.0, 2), "mm (gripper frame)",
                          "SOLVED, wrist spread " + fixed(wrist_spread, 1) + " deg" if auto_off else "given")
                    if sim.found:
                        print(camera_pose_vs_sim(sim, fit.rot, fit.trans))
                except e:
                    status = String("COULD NOT SAVE: ") + String(e)
            if not enough:
                ig_text_disabled(
                    String("save needs ") + String(min_poses) + "+ poses"
                )
            ig_separator()
            ig_text(status)
            ig_end()

            # ── the view ───────────────────────────────────────────────────
            _ = ig_begin_window(
                String("view"), 340.0, 10.0, Float32(fw) + 20.0,
                Float32(fh) + 60.0,
            )
            tex.image(Float32(fw), Float32(fh))
            var rect = ig_last_item_rect()
            for m in range(n_markers):
                # ⚠ THE PICKED MARKER IS GREEN AND EVERY OTHER ONE IS AMBER,
                # because "which tag is it using" is the question a wrong fit
                # makes you ask afterwards.
                var col = UInt32(0xFF00FF00) if m == pick else UInt32(0xFF20A0FF)
                for c in range(4):
                    var c2 = (c + 1) % 4
                    ig_overlay_line(
                        rect[0] + corners[m * 8 + c * 2],
                        rect[1] + corners[m * 8 + c * 2 + 1],
                        rect[0] + corners[m * 8 + c2 * 2],
                        rect[1] + corners[m * 8 + c2 * 2 + 1],
                        col,
                        2.0,
                    )
            ig_end()

            r.begin_frame()
            r.end_frame()
    finally:
        try:
            arm.set_torque(False)
        except:
            print("⚠ COULD NOT RELEASE TORQUE — run `pixi run soarm-torque-off`")
        det.close()
        tex.close()
        cap.close()

    if have_fit:
        print("")
        print(String(fit))
        if sim.found:
            print(camera_pose_vs_sim(sim, fit.rot, fit.trans))
        print("saved to", calib_path, "if you pressed save")
