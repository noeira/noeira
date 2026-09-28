"""The fisheye -> sim-pinhole path (`vision/fisheye.mojo`), against OpenCV and
against the tracer.

  1. `FisheyeLens.project` == `cv::fisheye::projectPoints`, rays out to 85 deg.
  2. `UndistortMap` == `cv::fisheye::initUndistortRectifyMap` (P = the sim
     pinhole), every output pixel of 640x480.
  3. `Pinhole.sim` == the tracer's `camera_sample_ray` — the ray through a
     pixel centre is the same ray in both, so an undistorted real frame and a
     rendered one put a 3D point at the same pixel.
  4. A SYNTHETIC CALIBRATION RECOVERS THE LENS: ChArUco corners (the studio's
     5x7 board) from 60 random poses at 12-37 cm through a known 170-degree lens, 0.2 px
     noise, `fisheye_calibrate`. What is checked is what matters downstream —
     the fitted undistortion map against the true one over the sim's 73.74
     deg field — plus the control that the lens's distortion terms move that
     map by many pixels (else "recovered" would be vacuous).
  5. `n_outside` is 0 at the rig's fovy and not at a fovy past the lens.
  6. `apply_hwc` on a linear ramp returns the map's own coordinate.
  7. `unproject` inverts `project` (the extrinsics tool's corner path).

Needs the OpenCV shim (`pixi run build-opencv`); no camera, no fixture.
"""

from std.math import sqrt, cos, sin, pi, log
from std.sys import exit

from noeira.math3d import Vec3 as Vec3Generic
from noeira.physics3d.raytrace.camera import CameraFrame, camera_sample_ray
from noeira.vision.opencv import (
    opencv_shim_available, fisheye_calibrate, fisheye_project,
    fisheye_undistort_map,
)
from noeira.vision.fisheye import FisheyeLens, Pinhole, UndistortMap
from noeira.vision.fisheye_calib import CalibViews, calibrate_fisheye, undistort_spread

comptime W = 640
comptime H = 480
comptime FOVY = 73.7398
"""The rig's `overhead_cam` / `wrist_cam` fovy (`so101_tower_stand.xml`)."""


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
        var a = self.u() + 1e-12
        var b = self.u()
        return sqrt(-2.0 * log(a)) * cos(2.0 * pi * b)


def rodrigues(rx: Float64, ry: Float64, rz: Float64) -> List[Float64]:
    """Row-major R of the axis-angle vector (OpenCV's `rvec`)."""
    var th = sqrt(rx * rx + ry * ry + rz * rz)
    if th < 1e-15:
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    var kx = rx / th
    var ky = ry / th
    var kz = rz / th
    var c = cos(th)
    var s = sin(th)
    var v = 1.0 - c
    return [
        c + kx * kx * v, kx * ky * v - kz * s, kx * kz * v + ky * s,
        ky * kx * v + kz * s, c + ky * ky * v, ky * kz * v - kx * s,
        kz * kx * v - ky * s, kz * ky * v + kx * s, c + kz * kz * v,
    ]


def true_lens() -> FisheyeLens:
    """A 170-degree module at 640x480: equidistant f ~ 320 px / 1.48 rad,
    principal point off centre, four non-trivial terms."""
    return FisheyeLens(
        216.4, 215.1, 321.3, 238.7, -0.06, 0.012, -0.004, 0.0006, W, H
    )


def main() raises:
    var fails = 0
    print("fisheye -> sim pinhole gate")
    if not opencv_shim_available():
        print("  the OpenCV shim is not built — `pixi run build-opencv`")
        print("=== NOT RUN (this is not a pass) ===")
        exit(1)

    var lens = true_lens()
    var pin = Pinhole.sim(FOVY, W, H)

    # ── 1. projection vs cv::fisheye::projectPoints ─────────────────────
    var rng = Rng(7)
    var pts = List[Float64]()
    for _ in range(500):
        var th = rng.u() * 85.0 * pi / 180.0
        var ph = rng.u() * 2.0 * pi
        pts.append(sin(th) * cos(ph))
        pts.append(sin(th) * sin(ph))
        pts.append(cos(th))
    var zero3: List[Float64] = [0.0, 0.0, 0.0]
    var cv_xy = List[Float64]()
    fisheye_project(pts, zero3, zero3, lens.k_matrix(), lens.d_vector(), cv_xy)
    var dp = 0.0
    for i in range(500):
        var uv = lens.project(pts[i * 3] / pts[i * 3 + 2], pts[i * 3 + 1] / pts[i * 3 + 2])
        dp = max(dp, max(abs(uv[0] - cv_xy[i * 2]), abs(uv[1] - cv_xy[i * 2 + 1])))
    check(fails, "1 project == cv::fisheye::projectPoints (rays to 85 deg)",
          dp < 1e-6, "max |d| " + String(dp) + " px")

    # ── 2. the map vs cv::fisheye::initUndistortRectifyMap ──────────────
    var um = UndistortMap(lens, pin)
    var mx = List[Float32]()
    var my = List[Float32]()
    fisheye_undistort_map(lens.k_matrix(), lens.d_vector(), pin.k_matrix(), W, H, mx, my)
    var dm = 0.0
    for i in range(W * H):
        dm = max(dm, max(abs(Float64(um.map_x[i]) - Float64(mx[i])),
                         abs(Float64(um.map_y[i]) - Float64(my[i]))))
    check(fails, "2 UndistortMap == cv::fisheye::initUndistortRectifyMap",
          dm < 2e-3, "max |d| " + String(dm) + " px over " + String(W * H))

    # ── 3. Pinhole.sim vs the tracer's own ray ──────────────────────────
    var thf = Float64(sin(0.5 * FOVY * pi / 180.0) / cos(0.5 * FOVY * pi / 180.0))
    var frame = CameraFrame[DType.float64](
        Vec3Generic[DType.float64](0, 0, 0),
        Vec3Generic[DType.float64](1, 0, 0),
        Vec3Generic[DType.float64](0, 1, 0),
        Vec3Generic[DType.float64](0, 0, 1),
        Scalar[DType.float64](FOVY),
        Scalar[DType.float64](thf),
    )
    var dr = 0.0
    for py in [0, 1, 117, 239, 240, 478, 479]:
        for px in [0, 1, 200, 319, 320, 638, 639]:
            var d = camera_sample_ray[DType.float64](frame, W, H, px, py, 0.0, 0.0)
            # tracer camera: looks down -Z, +Y up. OpenCV: +Z forward, +Y down.
            var a_t = Float64(d.x) / Float64(-d.z)
            var b_t = -Float64(d.y) / Float64(-d.z)
            var a_p = (Float64(px) - pin.cx) / pin.fx
            var b_p = (Float64(py) - pin.cy) / pin.fy
            dr = max(dr, max(abs(a_t - a_p), abs(b_t - b_p)))
    check(fails, "3 Pinhole.sim == raytrace camera_sample_ray (pixel centres)",
          dr < 1e-12, "max |d| " + String(dr) + " (normalised coords)")

    # ── 4. a synthetic calibration recovers the lens ────────────────────
    # The studio's board: 5x7 squares of 30 mm -> 4x6 = 24 inner corners.
    var board = List[Float64]()
    for j in range(6):
        for i in range(4):
            board.append(Float64(i + 1) * 0.03 - 0.075)
            board.append(Float64(j + 1) * 0.03 - 0.105)
            board.append(0.0)
    var views_in = CalibViews()
    var rng2 = Rng(2026)
    var views = 0
    while views < 60:
        var dist = 0.12 + rng2.u() * 0.25
        # aim anywhere in the image, out to 70 deg off axis: the edges are
        # where a fisheye's terms live
        var off = rng2.u() * 70.0 * pi / 180.0
        var az = rng2.u() * 2.0 * pi
        var cx = dist * sin(off) * cos(az)
        var cy = dist * sin(off) * sin(az)
        var cz = dist * cos(off)
        var R = rodrigues(rng2.sym(0.7), rng2.sym(0.7), rng2.sym(pi))
        var n = 0
        var vo = List[Float64]()
        var vi = List[Float64]()
        for p in range(24):
            var bx = board[p * 3]
            var by = board[p * 3 + 1]
            var x = R[0] * bx + R[1] * by + cx
            var y = R[3] * bx + R[4] * by + cy
            var z = R[6] * bx + R[7] * by + cz
            if z < 0.02:
                continue
            var uv = lens.project(x / z, y / z)
            var u = uv[0] + 0.2 * rng2.gauss()
            var v = uv[1] + 0.2 * rng2.gauss()
            if u < 2.0 or v < 2.0 or u > Float64(W - 3) or v > Float64(H - 3):
                continue
            vo.append(bx); vo.append(by); vo.append(0.0)
            vi.append(u); vi.append(v)
            n += 1
        if n < 12:
            continue
        views_in.add(vo^, vi^)
        views += 1
    # through `calibrate_fisheye` — the tool's path: a single
    # `fisheye_calibrate` over these views ABORTS (InitExtrinsics), see its header
    var res = calibrate_fisheye(views_in, W, H)
    var rms = res.rms
    var fit = res.lens
    print("    fit:", String(fit), "rms", rms, "| used", len(res.used),
          "dropped", len(res.dropped))
    check(fails, "4a the fit's rms is the injected noise", rms < 0.4,
          "rms " + String(rms) + " px (noise 0.2 px)")
    # ⚠ THE BOUND IS A SPEC, AND WHERE IT COMES FROM: the student sees the
    # frame at 320x240, so 1.5 px here is 0.75 px there; and the render-time
    # camera jitter it is trained under (`randomize.mojo` full: +-2 deg) is
    # ~13 px at this resolution. The RMS over the field is the typical error,
    # the max sits at the pinhole's corners, where the board's corners are
    # sparsest.
    var um_fit = UndistortMap(fit, pin)
    var dmap = 0.0
    var se = 0.0
    for i in range(W * H):
        var ex = Float64(um_fit.map_x[i]) - Float64(um.map_x[i])
        var ey = Float64(um_fit.map_y[i]) - Float64(um.map_y[i])
        dmap = max(dmap, max(abs(ex), abs(ey)))
        se += ex * ex + ey * ey
    var drms = sqrt(se / Float64(W * H))
    check(fails, "4b fitted undistortion == true over the sim's field (rms)",
          drms < 0.5, "rms " + String(drms) + " px")
    check(fails, "4b' ... and at its worst pixel", dmap < 1.5,
          "max " + String(dmap) + " px")
    # the tool's error bar must be of the order of the error it cannot see
    var spread = undistort_spread(views_in, res, pin)
    check(fails, "4d the bootstrap spread tracks the true map error",
          spread > drms / 3.0 and spread < drms * 3.0,
          "spread " + String(spread) + " px vs true rms " + String(drms))
    var nodist = FisheyeLens(lens.fx, lens.fy, lens.cx, lens.cy, 0.0, 0.0, 0.0, 0.0, W, H)
    var um_nd = UndistortMap(nodist, pin)
    var dnd = 0.0
    for i in range(W * H):
        dnd = max(dnd, max(abs(Float64(um_nd.map_x[i]) - Float64(um.map_x[i])),
                           abs(Float64(um_nd.map_y[i]) - Float64(um.map_y[i]))))
    check(fails, "4c control: the lens's terms move the map (else 4b is vacuous)",
          dnd > 5.0 * dmap and dnd > 1.0, "max |d| without them " + String(dnd) + " px")

    # ── 5. out-of-lens pixels ───────────────────────────────────────────
    check(fails, "5a no out-of-lens pixel at the rig's fovy", um.n_outside == 0,
          String(um.n_outside))
    var wide = UndistortMap(lens, Pinhole.sim(150.0, W, H))
    check(fails, "5b control: a 150-deg pinhole does leave the lens",
          wide.n_outside > 0, String(wide.n_outside))

    # ── 6. the remap on a linear ramp ───────────────────────────────────
    var src = List[UInt8](length=W * H * 3, fill=0)
    for y in range(H):
        for x in range(W):
            for c in range(3):
                src[(y * W + x) * 3 + c] = UInt8(Int(Float64(x) / 2.51))
    var dst = List[UInt8](length=W * H * 3, fill=0)
    um.apply_hwc(src, 0, 3, dst, 0)
    var dramp = 0.0
    for i in range(0, W * H, 97):
        # an independent bilinear read of the SAME source values
        var sx = Float64(um.map_x[i])
        var x0 = Int(sx)
        var f = sx - Float64(x0)
        var s0 = Float64(Int(Float64(x0) / 2.51))
        var s1 = Float64(Int(Float64(min(x0 + 1, W - 1)) / 2.51))
        var want = (1.0 - f) * s0 + f * s1
        dramp = max(dramp, abs(Float64(Int(dst[i * 3])) - want))
    check(fails, "6 apply_hwc interpolates the source at the map's point",
          dramp <= 0.5 + 1e-6, "max |d| " + String(dramp) + " levels")

    # ── 7. unproject inverts project (the extrinsics' corner path) ──────
    var du = 0.0
    for i in range(500):
        var a = pts[i * 3] / pts[i * 3 + 2]
        var b = pts[i * 3 + 1] / pts[i * 3 + 2]
        var uv = lens.project(a, b)
        var ab = lens.unproject(uv[0], uv[1])
        du = max(du, max(abs(ab[0] - a), abs(ab[1] - b)) / max(1.0, sqrt(a * a + b * b)))
    check(fails, "7 unproject(project(ray)) == ray, out to 85 deg",
          du < 1e-9, "max relative |d| " + String(du))

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILED")
        exit(1)
