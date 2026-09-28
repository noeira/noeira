"""The overhead pose estimator LIVE on the rig's camera — and the ruler check.

    pixi run mojo run -I . examples/so101/tower_pose_live.mojo --camera 0
    pixi run mojo run -I . examples/so101/tower_pose_live.mojo \\
        --camera /dev/soarm_cam_overhead \\
        --measured "200,-100;250,0;200,150;300,-150" --seconds 300

WHAT IT IS FOR. `tower_pose_real_check.mojo` scores the estimator on
recordings, where nothing says where the brick really was. This is the
number that does: put the brick on spots MEASURED with a ruler and read the
estimate. `--measured "x,y;..."` is in mm, +x FORWARD (where the arm faces at
pan 0), +y to the robot's LEFT, from the SHOULDER-PAN AXIS by default
(`--measured-from pan`: the centre of the rotating shoulder, the one point of
the base a ruler can find) — the tool adds the axis' 38.8 mm offset from the
world origin (the base frame, `so_arm101_tower.xml` body `shoulder`).
`--measured-from base` takes world coordinates as they are. Measure to the
brick's CENTRE. Nothing to press — whenever the brick has
held still for `--still-s` seconds (estimates within 1.5 mm of their mean),
one STABLE line is printed with the averaged pose; with `--measured` it is
matched to the nearest measured spot and the error printed. Move the brick,
wait, read. The arm need not be powered.

Every estimate is also printed at `--print-hz` (brick x, y, yaw, coverage,
residual; bowl x, y) and, with `--log FILE`, written as CSV.

The camera thread delivers RGB24 at 640x480 (the calibration's size — refused
otherwise), newest frame first (`take_latest`: a stale frame is a stale
pose). The lens is `--calib` (default the overhead calibration); the pose is
the asset's `overhead_cam`, or `--extrinsics FILE` from
`tower_pose_arm_calib.mojo` (`tasks/so101_tower_overhead.mojo`).

Colours default to the PRINTED props as measured on `cube-in-bowl-printed`
(bowl hue 36, brick 205 — the real bowl is ~11 deg more orange than the sim
material); pass `--brick-hsv` / `--bowl-hsv` to change them. The ROI is the
real desk's (|y| <= 0.28: its +y edge is at ~0.29 and the floor past it is
blue).
"""

from std.math import pi, sqrt, atan2, sin, cos
from std.sys import argv, exit
from std.time import perf_counter_ns

from noeira.core.concurrent.thread import sleep_us
from noeira.math3d import Vec3 as Vec3Generic
from noeira.tasks.so101_tower_overhead import (
    OVERHEAD_CALIB, DESK_Z, tower_overhead_pose, tower_desk_roi,
    printed_brick_hsv, printed_bowl_hsv, pose_confident,
)
from noeira.utils.fmt import fixed
from noeira.vision.calib_file import read_calib
from noeira.vision.camera_thread import CameraReader
from noeira.vision.fisheye import FisheyeLens
from noeira.vision.opencv import opencv_shim_available
from noeira.vision.tabletop_pose import (
    RigCamera, ColorClass, PrismModel, DeskROI, PoseEstimate,
    estimate_prism_pose,
)

comptime STILL_MM = 1.5
comptime PAN_AXIS_X = 0.0388353
"""World x of the shoulder-pan axis (`so_arm101_tower.xml`, body `shoulder`
pos; the base frame is the world origin, `base_pos` z only)."""


def _floats(s: String) raises -> List[Float64]:
    var out = List[Float64]()
    for p in s.split(","):
        out.append(Float64(String(String(p).strip())))
    return out^


def _hsv_arg(s: String) raises -> ColorClass:
    var v = _floats(s)
    if len(v) != 5:
        raise Error("an HSV class is hue,tol,s_min,v_min,v_max; got " + s)
    return ColorClass(v[0], v[1], v[2], v[3], v[4])


def main() raises:
    var args = argv()
    var camera = String("")
    var calib_path = String(OVERHEAD_CALIB)
    var extr_path = String("")
    var seconds = 120.0
    var print_hz = 2.0
    var still_s = 1.0
    var log_path = String("")
    var measured_s = String("")
    var measured_from = String("pan")
    var cls_brick = printed_brick_hsv()
    var cls_bowl = printed_bowl_hsv()
    var roi = tower_desk_roi()
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if i + 1 >= len(args):
            raise Error("flag " + a + " needs a value")
        var v = String(args[i + 1])
        if a == "--camera":
            camera = v
        elif a == "--calib":
            calib_path = v
        elif a == "--extrinsics":
            extr_path = v
        elif a == "--seconds":
            seconds = Float64(v)
        elif a == "--print-hz":
            print_hz = Float64(v)
        elif a == "--still-s":
            still_s = Float64(v)
        elif a == "--log":
            log_path = v
        elif a == "--measured":
            measured_s = v
        elif a == "--measured-from":
            if v != "pan" and v != "base":
                raise Error("--measured-from is pan or base, got " + v)
            measured_from = v
        elif a == "--brick-hsv":
            cls_brick = _hsv_arg(v)
        elif a == "--bowl-hsv":
            cls_bowl = _hsv_arg(v)
        elif a == "--roi":
            var r = _floats(v)
            roi = DeskROI(DESK_Z, r[0], r[1], r[2], r[3])
        else:
            raise Error("unknown flag " + a)
        i += 2
    if camera == "":
        raise Error("--camera <index | /dev/... path | video file> is required")
    if not opencv_shim_available():
        raise Error("the OpenCV shim is not built: pixi run build-opencv")

    var mx = List[Float64]()
    var my = List[Float64]()
    if measured_s != "":
        for p in measured_s.split(";"):
            var xy = _floats(String(p))
            if len(xy) != 2:
                raise Error("--measured is x,y;x,y;... in world mm, got " + measured_s)
            var off = PAN_AXIS_X if measured_from == "pan" else 0.0
            mx.append(xy[0] / 1000.0 + off)
            my.append(xy[1] / 1000.0)
        print(
            "measured spots (world mm, from the", measured_from, "):",
            len(mx), "— the pan axis is at world x", PAN_AXIS_X * 1000.0,
        )

    var cal = read_calib(calib_path)
    cal.require_size(640, 480)
    var lens = FisheyeLens.from_calib(cal)
    var pose = tower_overhead_pose(extr_path)
    var cam = RigCamera(lens, pose.pos, pose.rot_mj)
    print("camera pose:", pose.source)
    var brick = PrismModel.tower_brick()
    var bowl = PrismModel.tower_bowl()
    print("lens", lens)
    print("brick hsv", cls_brick, "| bowl hsv", cls_bowl, "| roi", roi)

    var reader = CameraReader.from_spec(camera, 640, 480, 30.0, rgb=True)
    reader.start()
    var frame = List[UInt8](length=reader.frame_bytes(), fill=UInt8(0))
    if reader.frame_bytes() != 640 * 480 * 3:
        raise Error("camera delivers " + String(reader.frame_bytes()) + " bytes, not 640x480x3")

    var log = String("t_s,brick_found,brick_x,brick_y,brick_yaw_deg,brick_cov,brick_res,bowl_found,bowl_x,bowl_y\n")
    var t0 = perf_counter_ns()
    var last_print = -1.0e9
    # the still window: recent confident brick estimates (time, x, y, yaw)
    var wt = List[Float64]()
    var wx = List[Float64]()
    var wy = List[Float64]()
    var wyaw = List[Float64]()
    var reported = False
    var n_est = 0
    var est_ms = 0.0
    var n_stable = 0
    var sum_err = 0.0
    var max_err = 0.0
    while True:
        var now = Float64(perf_counter_ns() - t0) * 1e-9
        if now > seconds:
            break
        if reader.take_latest(frame) == 0:
            _ = sleep_us(2000)
            continue
        var e0 = perf_counter_ns()
        var eb = estimate_prism_pose(frame, cam, cls_brick, brick, roi)
        var eo = estimate_prism_pose(frame, cam, cls_bowl, bowl, roi)
        est_ms += Float64(perf_counter_ns() - e0) * 1e-6
        n_est += 1
        if log_path != "":
            log += (
                fixed(now, 3) + "," + String(Int(eb.found)) + "," + fixed(eb.x, 5)
                + "," + fixed(eb.y, 5) + "," + fixed(eb.yaw * 180.0 / pi, 2) + ","
                + fixed(eb.coverage, 3) + "," + fixed(eb.residual, 3) + ","
                + String(Int(eo.found)) + "," + fixed(eo.x, 5) + "," + fixed(eo.y, 5)
                + "\n"
            )
        if now - last_print >= 1.0 / print_hz:
            last_print = now
            var s = String("t ") + fixed(now, 1) + "s  brick "
            if eb.found:
                s += (
                    "(" + fixed(eb.x * 1000.0, 1) + ", " + fixed(eb.y * 1000.0, 1)
                    + ") mm yaw " + fixed(eb.yaw * 180.0 / pi, 1) + " cov "
                    + fixed(eb.coverage, 2) + " res " + fixed(eb.residual, 2)
                )
            else:
                s += "-"
            s += "  | bowl "
            if eo.found:
                s += "(" + fixed(eo.x * 1000.0, 1) + ", " + fixed(eo.y * 1000.0, 1) + ") mm"
            else:
                s += "-"
            s += "  | " + fixed(est_ms / Float64(n_est), 1) + " ms/frame"
            print(s)

        # the still detector
        if not pose_confident(eb):
            continue
        wt.append(now)
        wx.append(eb.x)
        wy.append(eb.y)
        wyaw.append(eb.yaw)
        while len(wt) > 0 and wt[0] < now - still_s:
            _ = wt.pop(0)
            _ = wx.pop(0)
            _ = wy.pop(0)
            _ = wyaw.pop(0)
        var n = len(wx)
        var ax = 0.0
        var ay = 0.0
        for k in range(n):
            ax += wx[k]
            ay += wy[k]
        ax /= Float64(n)
        ay /= Float64(n)
        var spread = 0.0
        for k in range(n):
            spread = max(spread, sqrt((wx[k] - ax) ** 2 + (wy[k] - ay) ** 2) * 1000.0)
        var still = spread <= STILL_MM and n >= 5 and (now - wt[0]) >= 0.8 * still_s
        if not still:
            reported = False
            continue
        if reported:
            continue
        reported = True
        n_stable += 1
        # yaw mean on the 4-fold circle
        var sc = 0.0
        var ss = 0.0
        for k in range(n):
            sc += cos(4.0 * wyaw[k])
            ss += sin(4.0 * wyaw[k])
        var ayaw = atan2(ss, sc) / 4.0
        if ayaw < 0.0:
            ayaw += brick.period
        var line = (
            String("STABLE #") + String(n_stable) + "  brick (" + fixed(ax * 1000.0, 1)
            + ", " + fixed(ay * 1000.0, 1) + ") mm  yaw " + fixed(ayaw * 180.0 / pi, 1)
            + " deg  over " + String(n) + " frames, spread " + fixed(spread, 2) + " mm"
        )
        if len(mx) > 0:
            var best = 0
            var bd = 1.0e30
            for k in range(len(mx)):
                var d = sqrt((mx[k] - ax) ** 2 + (my[k] - ay) ** 2)
                if d < bd:
                    bd = d
                    best = k
            var ex = (ax - mx[best]) * 1000.0
            var ey = (ay - my[best]) * 1000.0
            line += (
                "  | vs measured (" + fixed(mx[best] * 1000.0, 0) + ", "
                + fixed(my[best] * 1000.0, 0) + "): err " + fixed(bd * 1000.0, 1)
                + " mm (dx " + fixed(ex, 1) + ", dy " + fixed(ey, 1) + ")"
            )
            sum_err += bd * 1000.0
            max_err = max(max_err, bd * 1000.0)
        print(line)

    reader.stop()
    if log_path != "":
        with open(log_path, "w") as f:
            f.write(log)
        print("wrote", log_path)
    print(
        "estimated", n_est, "frames,", fixed(est_ms / Float64(max(1, n_est)), 1),
        "ms/frame for brick + bowl;", n_stable, "stable placements",
    )
    if len(mx) > 0 and n_stable > 0:
        print(
            "vs measured: mean", fixed(sum_err / Float64(n_stable), 1), "mm, max",
            fixed(max_err, 1), "mm",
        )
