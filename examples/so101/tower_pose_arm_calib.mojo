"""Calibrate the overhead pose reader against the ARM: the brick where the
camera says, the fixed jaw's tip where the arm says, and the fit between them.

    pixi run -e jetson mojo run -I . examples/so101/tower_pose_arm_calib.mojo \\
        --camera /dev/soarm_cam_overhead --pairs 8

WHY THE ARM AND NOT A RULER. The ruler check (`tower_pose_live.mojo
--measured`, 2026-09-24) put the estimator 6.6 mm mean / 13 mm max from the
tape marks, repeatable to 2 mm, with a shared offset of (-4.7, -4.1) mm. That
offset is the camera pose OR the ruler's origin (the pan axis is inside the
servo) — and neither is the frame that matters. What matters is where the ARM
thinks the brick is, because the arm is what will grasp it. So the reference
here is the follower's own FK, with the joint zero the rig uses (`follower`).

## THIS PROGRAM NEVER ENERGISES THE ARM

Torque is released at start and again in a `finally` (if it dies hard:
`pixi run soarm-torque-off`). You move the follower BY HAND.

## The protocol — nothing to press

For each of `--pairs` places across the desk:
1. BRICK: put the brick down, arm out of the way; hold still. When the brick
   has been seen, confidently and still, for a second the tool prints
   `brick seen at (x, y)`.
2. TOUCH: move the follower by hand so the TIP OF THE FIXED JAW (the jaw that
   does not move) rests on the CENTRE of the brick's top face, the gripper
   roughly VERTICAL (the wrist zeros are unmeasured; holding the gripper the
   same way at every place keeps what they do constant). Hold still ~1 s. The
   tool prints the pair.
3. MOVE: take the arm away, move the brick to the next place. The next BRICK
   step starts as soon as the brick is seen still, somewhere else.
Spread the places over the whole desk: near and far, left and right. The
brick is seen with the arm AWAY — during the touch the gripper hides it.

## The fit and what it writes

A 2D rigid transform (a rotation about the vertical and a horizontal shift)
from the camera's brick positions to the jaw tip's, least squares. Applied to
the CAMERA it moves every desk-plane estimate by exactly that transform (a
horizontal ray-plane intersection commutes with a rotation about z and a
horizontal shift), so it is a pose correction, not a post-hoc patch. Printed:
the residual before and after, and the LEAVE-ONE-OUT residual (each pair
predicted by a fit to the others: what to expect on a new placement).
Written: `--out` (default `projects/so101-tower/cameras/camera_overhead_armcal.txt`),
the lens file's intrinsics plus the corrected pose as extrinsics, named
`overhead_armcal`, which `tower_pose_live` / `tower_pose_real_check` load with
`--extrinsics`; and `<out>.pairs.txt` (every pair, raw ticks, joint values)
for offline refits.

A residual that stays above a few mm after the fit with a pattern across the
desk (not a constant) means the rigid model is too small: camera height or
tilt, or the wrist zeros. Then the pairs file is what to refit from.
"""

from std.math import atan2, cos, sin, sqrt, pi
from std.sys import argv, exit
from std.time import perf_counter_ns

from noeira.core.concurrent.thread import sleep_us
from noeira.math3d import Mat3 as Mat3Generic, Vec3 as Vec3Generic
from noeira.robot.so101 import SO101Arm, SO101_N
from noeira.robot.so101.ports import follower_port
from noeira.robot.so101.sim_map import SimJointMap
from noeira.tasks.so101_tower_overhead import (
    OVERHEAD_CALIB, ARMCAL_SUFFIX, tower_overhead_pose, tower_desk_roi,
    printed_brick_hsv, pose_confident, TowerArmFK,
)
from noeira.utils.fmt import fixed
from noeira.vision.calib_file import read_calib, write_calib
from noeira.vision.camera_thread import CameraReader
from noeira.vision.fisheye import FisheyeLens
from noeira.vision.opencv import opencv_shim_available
from noeira.vision.tabletop_pose import RigCamera, PrismModel, estimate_prism_pose

comptime Vec3d = Vec3Generic[DType.float64]
comptime Mat3d = Mat3Generic[DType.float64]

comptime STILL_MM = 1.5
comptime STILL_TICKS_EPS = 4
"""A released servo jitters 1-2 ticks forever (the extrinsics tool's note)."""
comptime TOUCH_RADIUS = 0.04
"""The jaw tip must be within this of the brick's estimate (xy) to count."""
comptime TOUCH_MAX_Z = 0.07
"""... and below this height (world): the brick's top is at 0.027."""
comptime TIP_STILL = 0.0015
"""The jaw tip stays within this of its 1 s mean (3D) to be captured."""
comptime MOVED = 0.03
"""A new brick place is this far from the last one."""


def _rigid2d(
    ex: List[Float64], ey: List[Float64], tx: List[Float64], ty: List[Float64],
    skip: Int = -1,
) -> Tuple[Float64, Float64, Float64]:
    """(theta, dx, dy) with t = R(theta) e + d, least squares, pair `skip`
    left out."""
    var n = 0
    var mex = 0.0
    var mey = 0.0
    var mtx = 0.0
    var mty = 0.0
    for i in range(len(ex)):
        if i == skip:
            continue
        mex += ex[i]
        mey += ey[i]
        mtx += tx[i]
        mty += ty[i]
        n += 1
    mex /= Float64(n)
    mey /= Float64(n)
    mtx /= Float64(n)
    mty /= Float64(n)
    var sc = 0.0
    var ss = 0.0
    for i in range(len(ex)):
        if i == skip:
            continue
        var ax = ex[i] - mex
        var ay = ey[i] - mey
        var bx = tx[i] - mtx
        var by = ty[i] - mty
        sc += ax * bx + ay * by
        ss += ax * by - ay * bx
    var th = atan2(ss, sc)
    var c = cos(th)
    var s = sin(th)
    return (th, mtx - (c * mex - s * mey), mty - (s * mex + c * mey))


def _apply(f: Tuple[Float64, Float64, Float64], x: Float64, y: Float64) -> Tuple[Float64, Float64]:
    var c = cos(f[0])
    var s = sin(f[0])
    return (c * x - s * y + f[1], s * x + c * y + f[2])


def main() raises:
    var args = argv()
    var camera = String("")
    var calib_path = String(OVERHEAD_CALIB)
    var extr_in = String("")
    var port = String("")
    var n_pairs = 8
    var seconds = 900.0
    var out_path = String("projects/so101-tower/cameras/camera_overhead_armcal.txt")
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
            extr_in = v
        elif a == "--port":
            port = v
        elif a == "--pairs":
            n_pairs = Int(v)
        elif a == "--seconds":
            seconds = Float64(v)
        elif a == "--out":
            out_path = v
        else:
            raise Error("unknown flag " + a)
        i += 2
    if camera == "":
        raise Error("--camera <index | /dev/... path> is required")
    if not opencv_shim_available():
        raise Error("the OpenCV shim is not built: pixi run build-opencv")

    # ── camera (the pose this run CORRECTS: the asset's, or an earlier armcal)
    var lens_cal = read_calib(calib_path)
    lens_cal.require_size(640, 480)
    var lens = FisheyeLens.from_calib(lens_cal)
    var pose = tower_overhead_pose(extr_in)
    var cam = RigCamera(lens, pose.pos, pose.rot_mj)
    var roi = tower_desk_roi()
    var cls = printed_brick_hsv()
    var brick = PrismModel.tower_brick()
    print("camera pose:", pose.source)

    # ── arm, released; FK of the fixed jaw's tip ─────────────────────────
    var fk = TowerArmFK()
    var tip = fk.site_index("gripperframe")
    var lo = Array[Float64, SO101_N](fill=0.0)
    var hi = Array[Float64, SO101_N](fill=0.0)
    for k in range(SO101_N):
        lo[k] = fk.lo[k]
        hi[k] = fk.hi[k]
    var the_port = follower_port(port)
    print("opening", the_port, "...")
    var arm = SO101Arm(the_port, max_step_ticks=0)
    arm.bus.timeout_ms = 20
    arm.set_torque(False)
    print("  torque RELEASED — move the follower by hand")
    var jmap = SimJointMap.tower_follower(arm.cal, lo^, hi^)
    print("  " + jmap.describe())

    var reader = CameraReader.from_spec(camera, 640, 480, 30.0, rgb=True)
    reader.start()
    if reader.frame_bytes() != 640 * 480 * 3:
        raise Error("camera delivers " + String(reader.frame_bytes()) + " bytes, not 640x480x3")
    var frame = List[UInt8](length=reader.frame_bytes(), fill=UInt8(0))

    var raw = List[Int32](length=SO101_N, fill=Int32(0))
    var prev_raw = List[Int32](length=SO101_N, fill=Int32(0))
    var q = List[Float64](length=6, fill=0.0)
    var still_since = -1.0

    # pairs
    var ex = List[Float64]()
    var ey = List[Float64]()
    var tx = List[Float64]()
    var ty = List[Float64]()
    var tz = List[Float64]()
    var log = String(
        "# est_x est_y est_yaw_deg | tip_x tip_y tip_z (1 s mean) | q0..q5 (model rad, same window's mean) | raw0..raw5 (last sample)\n"
    )

    # state: 0 BRICK, 1 TOUCH, 2 MOVE
    var state = 0
    var bx = 0.0
    var by = 0.0
    var byaw = 0.0
    var wt = List[Float64]()
    var wx = List[Float64]()
    var wy = List[Float64]()
    var wyaw = List[Float64]()
    var tip_acc = List[Vec3d]()
    var tip_t = List[Float64]()
    var q_acc = List[List[Float64]]()
    var t0 = perf_counter_ns()
    print("\n[pair 1/", n_pairs, "] BRICK: put the brick down, arm out of the way, hands off")
    try:
        while len(ex) < n_pairs:
            var now = Float64(perf_counter_ns() - t0) * 1e-9
            if now > seconds:
                print("time is up")
                break
            # ── the arm ──────────────────────────────────────────────────
            var arm_ok = arm.read_positions(Span(raw)) == SO101_N
            var tip_p = Vec3d.zero()
            if arm_ok:
                var moved = 0
                for k in range(SO101_N):
                    moved = max(moved, abs(Int(raw[k]) - Int(prev_raw[k])))
                    prev_raw[k] = raw[k]
                if moved <= STILL_TICKS_EPS:
                    if still_since < 0.0:
                        still_since = now
                else:
                    still_since = -1.0
                for k in range(6):
                    q[k] = jmap.to_sim_unclamped(arm.cal, k, raw[k])
                fk.set_qpos(q)
                tip_p = fk.site_pos(tip)
            else:
                still_since = -1.0
            var arm_still_s = now - still_since if still_since >= 0.0 else 0.0

            # ── the camera ───────────────────────────────────────────────
            var got = reader.take_latest(frame) > 0
            if not got:
                _ = sleep_us(2000)
                continue
            var e = estimate_prism_pose(frame, cam, cls, brick, roi)
            var conf = pose_confident(e)

            if state == 0 or state == 2:
                if not conf:
                    wt.clear()
                    wx.clear()
                    wy.clear()
                    wyaw.clear()
                    continue
                if state == 2 and sqrt((e.x - bx) ** 2 + (e.y - by) ** 2) < MOVED:
                    continue
                wt.append(now)
                wx.append(e.x)
                wy.append(e.y)
                wyaw.append(e.yaw)
                while len(wt) > 0 and wt[0] < now - 1.0:
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
                if n >= 10 and now - wt[0] >= 0.8 and spread <= STILL_MM:
                    bx = ax
                    by = ay
                    byaw = wyaw[n - 1]
                    state = 1
                    tip_acc.clear()
                    print(
                        "  brick seen at (", fixed(bx * 1000.0, 1), ",", fixed(by * 1000.0, 1),
                        ") mm world, spread", fixed(spread, 1),
                        "mm\n  TOUCH: put the FIXED jaw's tip on the CENTRE of the brick's"
                        " top face, gripper roughly vertical, and hold still",
                    )
                    wt.clear()
                    wx.clear()
                    wy.clear()
                    wyaw.clear()
                continue

            # state 1: TOUCH
            if not arm_ok:
                continue
            var dxy = sqrt((tip_p.x - bx) ** 2 + (tip_p.y - by) ** 2)
            if dxy > TOUCH_RADIUS or Float64(tip_p.z) > TOUCH_MAX_Z:
                tip_acc.clear()
                tip_t.clear()
                q_acc.clear()
                continue
            # a sliding 1 s window of the TIP itself: the per-step tick test
            # above misses a slow drift (2026-09-24, pair 8 moved ~13 mm
            # inside a "still" second), so stillness is judged on the tip
            tip_acc.append(tip_p)
            tip_t.append(now)
            q_acc.append(q.copy())
            while len(tip_t) > 0 and tip_t[0] < now - 1.0:
                _ = tip_acc.pop(0)
                _ = tip_t.pop(0)
                _ = q_acc.pop(0)
            var wmx = 0.0
            var wmy = 0.0
            var wmz = 0.0
            for p in tip_acc:
                wmx += Float64(p.x)
                wmy += Float64(p.y)
                wmz += Float64(p.z)
            var wn = Float64(len(tip_acc))
            var drift = 0.0
            for p in tip_acc:
                drift = max(drift, sqrt(
                    (Float64(p.x) - wmx / wn) ** 2 + (Float64(p.y) - wmy / wn) ** 2
                    + (Float64(p.z) - wmz / wn) ** 2
                ))
            if now - tip_t[0] >= 0.9 and len(tip_acc) >= 10 and drift <= TIP_STILL:
                var mx = wmx / wn
                var my = wmy / wn
                var mz = wmz / wn
                # the joints logged are the SAME window's mean (they were the
                # last sample's, which disagreed with the averaged tip)
                var qm = List[Float64](length=6, fill=0.0)
                for qs in q_acc:
                    for k in range(6):
                        qm[k] += qs[k] / wn
                ex.append(bx)
                ey.append(by)
                tx.append(mx)
                ty.append(my)
                tz.append(mz)
                log += (
                    fixed(bx, 5) + " " + fixed(by, 5) + " " + fixed(byaw * 180.0 / pi, 2)
                    + " | " + fixed(mx, 5) + " " + fixed(my, 5) + " " + fixed(mz, 5) + " |"
                )
                for k in range(6):
                    log += " " + fixed(qm[k], 5)
                log += " |"
                for k in range(6):
                    log += " " + String(Int(raw[k]))
                log += "\n"
                var k_pair = len(ex)
                print(
                    "  PAIR", k_pair, ": camera (", fixed(bx * 1000.0, 1), ",",
                    fixed(by * 1000.0, 1), ") vs jaw tip (", fixed(mx * 1000.0, 1), ",",
                    fixed(my * 1000.0, 1), ", z", fixed(mz * 1000.0, 1), ") mm -> dx",
                    fixed((bx - mx) * 1000.0, 1), "dy", fixed((by - my) * 1000.0, 1),
                )
                if k_pair >= 3:
                    var fr = _rigid2d(ex, ey, tx, ty)
                    var r = 0.0
                    for j in range(k_pair):
                        var pp = _apply(fr, ex[j], ey[j])
                        r += (pp[0] - tx[j]) ** 2 + (pp[1] - ty[j]) ** 2
                    print(
                        "  running fit: rotation", fixed(fr[0] * 180.0 / pi, 2), "deg, shift (",
                        fixed(fr[1] * 1000.0, 1), ",", fixed(fr[2] * 1000.0, 1),
                        ") mm, rms after", fixed(sqrt(r / Float64(k_pair)) * 1000.0, 1), "mm",
                    )
                state = 2
                if k_pair < n_pairs:
                    print(
                        "\n[pair", k_pair + 1, "/", n_pairs, "] MOVE: arm away, brick to a new"
                        " place (spread them: near/far, left/right), hands off",
                    )
    finally:
        try:
            arm.set_torque(False)
        except:
            print("⚠ COULD NOT RELEASE TORQUE — run `pixi run soarm-torque-off`")
        reader.stop()

    var n = len(ex)
    if n < 3:
        print("only", n, "pairs: nothing fitted, nothing written")
        return

    # ── the fit ──────────────────────────────────────────────────────────
    var f = _rigid2d(ex, ey, tx, ty)
    var r0 = 0.0
    var r1 = 0.0
    var rl = 0.0
    var max1 = 0.0
    var maxl = 0.0
    for j in range(n):
        r0 += (ex[j] - tx[j]) ** 2 + (ey[j] - ty[j]) ** 2
        var p = _apply(f, ex[j], ey[j])
        var e1 = (p[0] - tx[j]) ** 2 + (p[1] - ty[j]) ** 2
        r1 += e1
        max1 = max(max1, sqrt(e1))
        var fl = _rigid2d(ex, ey, tx, ty, skip=j)
        var pl = _apply(fl, ex[j], ey[j])
        var el = (pl[0] - tx[j]) ** 2 + (pl[1] - ty[j]) ** 2
        rl += el
        maxl = max(maxl, sqrt(el))
    var rms0 = sqrt(r0 / Float64(n)) * 1000.0
    var rms1 = sqrt(r1 / Float64(n)) * 1000.0
    var rmsl = sqrt(rl / Float64(n)) * 1000.0
    print("\n", n, "pairs")
    print("  before (", pose.source, "): rms", fixed(rms0, 1), "mm")
    print(
        "  fit: rotation", fixed(f[0] * 180.0 / pi, 2), "deg about world z, shift (",
        fixed(f[1] * 1000.0, 1), ",", fixed(f[2] * 1000.0, 1), ") mm",
    )
    print("  after: rms", fixed(rms1, 1), "mm, max", fixed(max1 * 1000.0, 1), "mm")
    print(
        "  leave-one-out (a NEW placement): rms", fixed(rmsl, 1), "mm, max",
        fixed(maxl * 1000.0, 1), "mm",
    )

    # ── the corrected camera: pos' = R pos + d, rot' = R rot ─────────────
    var c = cos(f[0])
    var s = sin(f[0])
    var R = Mat3d.from_cols(Vec3d(c, s, 0.0), Vec3d(-s, c, 0.0), Vec3d(0.0, 0.0, 1.0))
    var pos2 = R * pose.pos + Vec3d(f[1], f[2], 0.0)
    var rot2 = R @ pose.rot_mj
    var out = read_calib(calib_path)
    out.name = String("overhead") + ARMCAL_SUFFIX
    out.has_extrinsics = True
    out.rot = Mat3d.from_cols(rot2.col(0), -rot2.col(1), -rot2.col(2))
    out.trans = pos2 - pose.base_off
    out.rms_mm = rmsl
    out.poses = n
    write_calib(out_path, out)
    with open(out_path + ".pairs.txt", "w") as fh:
        fh.write(log)
    print("  wrote", out_path, "(extrinsics = the corrected pose; rms_mm = the leave-one-out)")
    print("  and", out_path + ".pairs.txt")
    print(
        "  use it: tower_pose_live / tower_pose_real_check --extrinsics", out_path,
    )
