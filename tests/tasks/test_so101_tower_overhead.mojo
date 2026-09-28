"""`tasks/so101_tower_overhead.mojo`: the overhead camera's pose sources and
the arm calibration's file round trip.

  1. No `--extrinsics`: the pose IS the asset's `overhead_cam`.
  2. The correction `tower_pose_arm_calib.mojo` writes — the camera moved by a
     rotation R about world z and a horizontal shift d, stored as calibration
     extrinsics (camera -> BASE, OpenCV axes, origin in the BASE frame) — read
     back through `tower_overhead_pose`, moves every desk-plane point by
     exactly R p + d. That is the claim the tool's fit rests on, and the file
     path crosses two frame changes (MuJoCo <-> OpenCV axes, world <-> base)
     where a sign or an offset could hide.
  3. CONTROL: the same test with the correction's rotation sign flipped in
     the written file fails by > 5 mm (the check is not vacuous).
  4. A calibration file whose name is not `*_armcal` (the lens file's own
     stale extrinsics) is REFUSED.
  5. `TowerArmFK.camera_pose` at the rest pose IS `tower_sim_camera`'s pose,
     for the overhead camera (on the stand) and the wrist camera (on the
     gripper); and at a bent arm the wrist camera moves WITH the gripper (its
     pose in the gripper body's frame unchanged) while the overhead one does
     not move. CONTROL: at the bent arm the wrist camera is > 50 mm from
     its rest pose (the check sees the arm move).
"""

from std.math import cos, sin, sqrt, pi
from std.sys import exit

from noeira.math3d import Mat3 as Mat3Generic, Vec3 as Vec3Generic
from noeira.tasks.so101_tower_camera_pose import tower_sim_camera
from noeira.tasks.so101_tower_overhead import tower_overhead_pose, ARMCAL_SUFFIX, DESK_Z, TowerArmFK
from noeira.vision.calib_file import CameraCalib, write_calib
from noeira.vision.fisheye import Pinhole
from noeira.vision.tabletop_pose import RigCamera

comptime Vec3d = Vec3Generic[DType.float64]
comptime Mat3d = Mat3Generic[DType.float64]


def fixed_mm(v: Float64) -> String:
    return String(v)


def check(mut fails: Int, name: String, ok: Bool, detail: String):
    if ok:
        print("  PASS  " + name + "  " + detail)
    else:
        fails += 1
        print("  FAIL  " + name + "  " + detail)


def write_armcal(path: String, name: String, th: Float64, d: Vec3d) raises:
    var sim = tower_sim_camera("overhead_cam")
    var c = cos(th)
    var s = sin(th)
    var R = Mat3d.from_cols(Vec3d(c, s, 0.0), Vec3d(-s, c, 0.0), Vec3d(0.0, 0.0, 1.0))
    var pos2 = R * sim.pos + d
    var rot2 = R @ sim.rot
    var cal = CameraCalib(name, -1, 640, 480, 320.0, 320.0, 319.5, 239.5)
    cal.has_extrinsics = True
    cal.rot = Mat3d.from_cols(rot2.col(0), -rot2.col(1), -rot2.col(2))
    cal.trans = pos2 - sim.base_off
    cal.poses = 5
    write_calib(path, cal)


def worst_mm(extr: String, th: Float64, d: Vec3d) raises -> Float64:
    var pin = Pinhole.sim(73.7398, 640, 480)
    var sim = tower_sim_camera("overhead_cam")
    var cam0 = RigCamera(pin, sim.pos, sim.rot)
    var p = tower_overhead_pose(extr)
    var cam1 = RigCamera(pin, p.pos, p.rot_mj)
    var c = cos(th)
    var s = sin(th)
    var worst = 0.0
    for v in range(40, 480, 80):
        for u in range(40, 640, 80):
            var a = cam0.plane_point(Float64(u), Float64(v), DESK_Z)
            var b = cam1.plane_point(Float64(u), Float64(v), DESK_Z)
            if not (a[2] and b[2]):
                continue
            var ex = c * a[0] - s * a[1] + Float64(d.x)
            var ey = s * a[0] + c * a[1] + Float64(d.y)
            worst = max(worst, sqrt((ex - b[0]) ** 2 + (ey - b[1]) ** 2) * 1000.0)
    return worst


def main() raises:
    var fails = 0
    var sim = tower_sim_camera("overhead_cam")
    var p0 = tower_overhead_pose()
    var dp = (p0.pos - sim.pos).length()
    check(fails, "no extrinsics = the asset", dp == 0.0 and p0.source.startswith("asset"), String(dp))

    var th = 0.9 * pi / 180.0
    var d = Vec3d(-0.0045, 0.0021, 0.0)
    var path = String("/tmp/test_so101_tower_overhead_armcal.txt")
    write_armcal(path, String("overhead") + ARMCAL_SUFFIX, th, d)
    var w = worst_mm(path, th, d)
    check(fails, "armcal moves desk points by R p + d", w < 1e-6, "worst " + String(w) + " mm")

    write_armcal(path, String("overhead") + ARMCAL_SUFFIX, -th, d)
    var wc = worst_mm(path, th, d)
    check(fails, "control: flipped rotation is caught", wc > 5.0, "worst " + String(wc) + " mm")

    write_armcal(path, String("overhead"), th, d)
    var refused = False
    try:
        _ = tower_overhead_pose(path)
    except:
        refused = True
    check(fails, "non-armcal extrinsics refused", refused, "")

    # 5. camera poses through the arm's FK
    var fk = TowerArmFK()
    var rest: List[Float64] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fk.set_qpos(rest)
    for name in ["overhead_cam", "wrist_cam"]:
        var sc = tower_sim_camera(name)
        var ci = fk.camera_index(name)
        var cp = fk.camera_pose(ci)
        var dp = (cp[0] - sc.pos).length() * 1000.0
        var dr = 0.0
        for k in range(3):
            dr = max(dr, (cp[1].col(k) - sc.rot.col(k)).length())
        check(fails, String("camera_pose at rest = tower_sim_camera ") + name, dp < 1e-9 and dr < 1e-12,
              fixed_mm(dp) + " mm, rot " + String(dr))
    var wi = fk.camera_index("wrist_cam")
    var oi = fk.camera_index("overhead_cam")
    var w0 = fk.camera_pose(wi)
    var o0 = fk.camera_pose(oi)
    var gs = fk.site_index("grasp_center")
    var g0p = fk.site_pos(gs)
    var g0r = fk.site_body_rot(gs)
    var bent: List[Float64] = [0.6, -0.4, 0.7, 0.5, 1.1, 0.3]
    fk.set_qpos(bent)
    var w1 = fk.camera_pose(wi)
    var o1 = fk.camera_pose(oi)
    var g1p = fk.site_pos(gs)
    var g1r = fk.site_body_rot(gs)
    # the wrist camera in the gripper body's frame, via `grasp_center` (on the
    # same body): R^T (cam - site) must not change
    var l0 = g0r.transpose() * (w0[0] - g0p)
    var l1 = g1r.transpose() * (w1[0] - g1p)
    var dl = (l1 - l0).length() * 1000.0
    check(fails, "the wrist camera rides the gripper", dl < 1e-9, fixed_mm(dl) + " mm in the gripper frame")
    var mo = (o1[0] - o0[0]).length() * 1000.0
    check(fails, "the overhead camera does not move", mo < 1e-9, fixed_mm(mo) + " mm")
    var mw = (w1[0] - w0[0]).length() * 1000.0
    check(fails, "control: the bent arm moves the wrist camera", mw > 50.0, fixed_mm(mw) + " mm")

    if fails > 0:
        print("FAILED:", fails)
        exit(1)
    print("ALL PASS")
