"""`fit_rigid_with_offset` — the marker offset solved with the camera pose.

Synthetic: a known camera (camera -> base R, T) and a marker at a known
offset on the gripper (the rig's camera-mount block, ~(0.0025, -0.067, 0.017)
m), seen at 15 gripper poses spread over the workspace with the wrist
turned (roll and pitch):

  1. noise-free: the offset and the camera come back to 1e-6;
  2. 1 mm of marker noise: offset within 3 mm, camera within 3 mm / 0.5 deg;
  3. control: the plain fit with offset = 0 on the same data is off by more
     than 20 mm — the offset is what the solve buys;
  4. degenerate: the wrist turning about ONE axis only is refused (the offset
     along that axis is undetermined), not answered.

Needs the OpenCV shim (`fit_rigid` uses its 3x3 SVD); no camera, no arm.
"""

from std.math import sqrt, cos, sin, log, pi, acos
from std.sys import exit

from noeira.math3d import Mat3 as Mat3Generic, Vec3 as Vec3Generic
from noeira.vision.extrinsics import fit_rigid, fit_rigid_with_offset
from noeira.vision.opencv import opencv_shim_available

comptime Vec3d = Vec3Generic[DType.float64]
comptime Mat3d = Mat3Generic[DType.float64]
comptime N = 15


def check(mut fails: Int, name: String, ok: Bool, detail: String):
    if ok:
        print("  PASS  " + name + "  " + detail)
    else:
        fails += 1
        print("  FAIL  " + name + "  " + detail)


struct Rng:
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def u(mut self) -> Float64:
        self.s += UInt64(0x9E3779B97F4A7C15)
        var z = self.s
        z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
        z = z ^ (z >> 31)
        return Float64(Int(z >> 11)) * (1.0 / 9007199254740992.0)

    def sym(mut self, r: Float64) -> Float64:
        return (2.0 * self.u() - 1.0) * r

    def gauss(mut self) -> Float64:
        return sqrt(-2.0 * log(self.u() + 1e-12)) * cos(2.0 * pi * self.u())


def rot_angle_deg(a: Mat3d, b: Mat3d) -> Float64:
    var c = (Float64((a.transpose() @ b).trace()) - 1.0) / 2.0
    c = 1.0 if c > 1.0 else (-1.0 if c < -1.0 else c)

    return acos(c) * 180.0 / pi


def make(
    mut rng: Rng, noise_m: Float64, one_axis: Bool, r_cam: Mat3d, t_cam: Vec3d,
    off: Vec3d, mut cam: List[Float64], mut gp: List[Float64], mut gr: List[Float64],
):
    cam = List[Float64]()
    gp = List[Float64]()
    gr = List[Float64]()
    for _ in range(N):
        var t = Vec3d(0.12 + rng.u() * 0.25, rng.sym(0.2), 0.05 + rng.u() * 0.2)
        var r: Mat3d
        if one_axis:
            r = Mat3d.rotation_z(rng.sym(1.2))
        else:
            r = Mat3d.rotation_z(rng.sym(1.2)) @ Mat3d.rotation_y(rng.sym(0.8)) @ Mat3d.rotation_x(rng.sym(0.8))
        var p_base = t + r * off
        # the camera sees it at R_cam^T (p - T)
        var c = r_cam.transpose() * (p_base - t_cam)
        cam.append(Float64(c.x) + noise_m * rng.gauss())
        cam.append(Float64(c.y) + noise_m * rng.gauss())
        cam.append(Float64(c.z) + noise_m * rng.gauss())
        gp.append(t.x); gp.append(t.y); gp.append(t.z)
        for i in range(3):
            for j in range(3):
                var row = r.row(i)
                gr.append(Float64(row.x if j == 0 else (row.y if j == 1 else row.z)))


def main() raises:
    var fails = 0
    print("extrinsics: the marker offset solved with the camera")
    if not opencv_shim_available():
        print("  the OpenCV shim is not built — `pixi run build-opencv`")
        exit(1)
    # a camera ~0.5 m above the workspace, looking down, tilted
    var r_cam = Mat3d.rotation_z(0.3) @ Mat3d.rotation_x(pi - 0.45)
    var t_cam = Vec3d(0.04, -0.04, 0.54)
    var off = Vec3d(0.0025, -0.067, 0.017)
    var cam = List[Float64]()
    var gp = List[Float64]()
    var gr = List[Float64]()

    # ── 1. noise-free ───────────────────────────────────────────────────
    var rng = Rng(11)
    make(rng, 0.0, False, r_cam, t_cam, off, cam, gp, gr)
    var f = fit_rigid_with_offset(cam, gp, gr)
    var eo = Float64((f.offset - off).length())
    var et = Float64((f.fit.trans - t_cam).length())
    check(fails, "1 noise-free: offset and camera exact",
          eo < 1e-6 and et < 1e-6 and rot_angle_deg(f.fit.rot, r_cam) < 1e-4,
          "offset " + String(eo * 1000.0) + " mm, camera " + String(et * 1000.0)
          + " mm, " + String(f.iterations) + " iterations, wrist spread "
          + String(f.rot_spread_deg) + " deg")

    # ── 2. 1 mm of marker noise ─────────────────────────────────────────
    var rng2 = Rng(12)
    make(rng2, 0.001, False, r_cam, t_cam, off, cam, gp, gr)
    var f2 = fit_rigid_with_offset(cam, gp, gr)
    var eo2 = Float64((f2.offset - off).length()) * 1000.0
    var et2 = Float64((f2.fit.trans - t_cam).length()) * 1000.0
    var er2 = rot_angle_deg(f2.fit.rot, r_cam)
    check(fails, "2 1 mm noise: offset < 3 mm, camera < 3 mm / 0.5 deg",
          eo2 < 3.0 and et2 < 3.0 and er2 < 0.5,
          "offset " + String(eo2) + " mm, camera " + String(et2) + " mm / "
          + String(er2) + " deg, rms " + String(f2.fit.rms_mm) + " mm")

    # ── 3. control: offset = 0 on the same data ─────────────────────────
    var base0 = List[Float64]()
    for k in range(N):
        base0.append(gp[k * 3]); base0.append(gp[k * 3 + 1]); base0.append(gp[k * 3 + 2])
    var f0 = fit_rigid(cam, base0)
    var et0 = Float64((f0.trans - t_cam).length()) * 1000.0
    check(fails, "3 control: a zero offset is > 20 mm off (or its rms says so)",
          et0 > 20.0 or f0.rms_mm > 20.0,
          "camera " + String(et0) + " mm, rms " + String(f0.rms_mm) + " mm")

    # ── 4. one wrist axis only: refused ─────────────────────────────────
    var rng4 = Rng(13)
    make(rng4, 0.0, True, r_cam, t_cam, off, cam, gp, gr)
    var refused = False
    try:
        _ = fit_rigid_with_offset(cam, gp, gr)
    except e:
        refused = String(e).find("barely rotated") >= 0
    check(fails, "4 rotation about one axis only is refused", refused, "")

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILED")
        exit(1)
