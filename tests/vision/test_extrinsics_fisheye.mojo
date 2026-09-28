"""The extrinsics path for the rig's fisheye cameras, and its comparison with the sim.

  1. THE SIM'S OWN POSE COMES BACK UNCHANGED. The tower's `overhead_cam`
     pose, converted to what the extrinsics fit produces (OpenCV axes,
     camera -> robot BASE), then through `pose_delta`: 0 mm, 0 deg, and the
     asset's own `<camera pos="0.05426 -0.0365 0.53585" xyaxes="0 -1 0
     0.9055 0 0.4243">` in the parent body frame. A flipped axis, a missing
     base offset or a wrong frame fails here.
  2. A KNOWN ERROR IS REPORTED AS ITSELF: 10 mm and 3 deg of perturbation
     read back as 10 mm and 3 deg (the positive control for 1).
  3. FISHEYE PnP: a 30 mm marker's corners projected through a 170-degree
     lens, `unproject`ed, `solve_pnp` with K = I recovers the marker's
     position to 1e-6 m. Control: solving on the RAW pixels as if the lens
     were a pinhole is off by more than 5 mm, so the undistortion matters.

Needs the OpenCV shim (3); no camera, no arm.
"""

from std.math import sqrt, cos, sin, pi
from std.sys import exit

from noeira.math3d import Mat3 as Mat3Generic, Vec3 as Vec3Generic
from noeira.tasks.so101_tower_camera_pose import (
    tower_sim_camera, pose_delta, fit_to_mujoco_rot,
)
from noeira.vision.fisheye import FisheyeLens
from noeira.vision.opencv import (
    opencv_shim_available, solve_pnp, SOLVEPNP_IPPE_SQUARE,
)

comptime Vec3d = Vec3Generic[DType.float64]
comptime Mat3d = Mat3Generic[DType.float64]


def check(mut fails: Int, name: String, ok: Bool, detail: String):
    if ok:
        print("  PASS  " + name + "  " + detail)
    else:
        fails += 1
        print("  FAIL  " + name + "  " + detail)


def dist3(a: Vec3d, b: Vec3d) -> Float64:
    return Float64((a - b).length())


def main() raises:
    var fails = 0
    print("extrinsics: fisheye PnP + comparison with the sim camera")

    # ── 1. the sim's own pose round-trips ───────────────────────────────
    var sim = tower_sim_camera(String("overhead_cam"))
    check(fails, "1a the tower scene has overhead_cam", sim.found, sim.name)
    # what the fit would report for a camera exactly where the sim's is:
    # R_cv = R_mj @ diag(1,-1,-1) (the conversion is its own inverse)
    var r_cv = fit_to_mujoco_rot(sim.rot)
    var t_base = sim.pos - sim.base_off
    var d = pose_delta(sim, r_cv, t_base)
    check(fails, "1b 0 mm, 0 deg for the sim's own pose",
          Float64(d.dpos_mm.length()) < 1e-6 and d.rot_deg < 1e-4 and d.axis_deg < 1e-4,
          String(Float64(d.dpos_mm.length())) + " mm, " + String(d.rot_deg) + " deg")
    var y_asset = Vec3d(0.9055, 0.0, 0.4243)
    y_asset = y_asset / y_asset.length()
    check(fails, "1c the asset's own <camera pos> comes back",
          dist3(d.pos_local, Vec3d(0.05426, -0.0365, 0.53585)) < 1e-9,
          String(d.pos_local.x) + " " + String(d.pos_local.y) + " " + String(d.pos_local.z))
    check(fails, "1d ... and its xyaxes",
          dist3(d.x_local, Vec3d(0.0, -1.0, 0.0)) < 1e-4 and dist3(d.y_local, y_asset) < 1e-4,
          "")
    check(fails, "1e the base offset is the family's (0, 0, 0.005)",
          dist3(sim.base_off, Vec3d(0.0, 0.0, 0.005)) < 1e-12, "")

    # ── 2. a known error reads back as itself ───────────────────────────
    var ax = Vec3d(0.3, -0.5, 0.8)
    ax = ax / ax.length()
    var rot3 = Mat3d.rotation_axis(ax, 3.0 * pi / 180.0)
    var r_pert = fit_to_mujoco_rot(sim.rot @ rot3)
    var d2 = pose_delta(sim, r_pert, t_base + Vec3d(0.006, -0.008, 0.0))
    check(fails, "2 a 10 mm / 3 deg perturbation reads 10 mm / 3 deg",
          abs(Float64(d2.dpos_mm.length()) - 10.0) < 1e-6 and abs(d2.rot_deg - 3.0) < 1e-6,
          String(Float64(d2.dpos_mm.length())) + " mm, " + String(d2.rot_deg) + " deg")

    # ── 3. fisheye PnP ──────────────────────────────────────────────────
    if not opencv_shim_available():
        print("  the OpenCV shim is not built — `pixi run build-opencv`")
        exit(1)
    var lens = FisheyeLens(277.07, 276.83, 317.69, 257.07, 0.0427, -0.0366, 0.0222, -0.0070, 640, 480)
    var half = 0.015
    var obj: List[Float64] = [-half, half, 0.0, half, half, 0.0, half, -half, 0.0, -half, -half, 0.0]
    var worst_fe = 0.0
    var worst_raw = 0.0
    for trial in range(6):
        # the marker 25-45 cm away, off axis by up to ~45 deg, tilted
        var t = Vec3d(
            0.12 * cos(Float64(trial)), 0.10 * sin(1.7 * Float64(trial)),
            0.25 + 0.04 * Float64(trial),
        )
        var R = Mat3d.rotation_axis(Vec3d(0.6, 0.8, 0.0), 0.35 * Float64(trial % 3))
        var fe_xy = List[Float64]()
        var raw_xy = List[Float64]()
        for c in range(4):
            var p = R * Vec3d(obj[c * 3], obj[c * 3 + 1], 0.0) + t
            var uv = lens.project(Float64(p.x) / Float64(p.z), Float64(p.y) / Float64(p.z))
            var ab = lens.unproject(uv[0], uv[1])
            fe_xy.append(ab[0]); fe_xy.append(ab[1])
            raw_xy.append(uv[0]); raw_xy.append(uv[1])
        var rv = List[Float64]()
        var tv = List[Float64]()
        solve_pnp(obj, fe_xy, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                  List[Float64](), rv, tv, SOLVEPNP_IPPE_SQUARE)
        worst_fe = max(worst_fe, dist3(Vec3d(tv[0], tv[1], tv[2]), t))
        solve_pnp(obj, raw_xy, lens.k_matrix(), List[Float64](), rv, tv, SOLVEPNP_IPPE_SQUARE)
        worst_raw = max(worst_raw, dist3(Vec3d(tv[0], tv[1], tv[2]), t))
    check(fails, "3a undistorted corners + K=I recover the marker",
          worst_fe < 1e-6, "worst " + String(worst_fe * 1000.0) + " mm")
    check(fails, "3b control: raw pixels as a pinhole are wrong",
          worst_raw > 0.005, "worst " + String(worst_raw * 1000.0) + " mm")

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILED")
        exit(1)
