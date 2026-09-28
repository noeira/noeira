"""The tabletop pose estimator (`vision/tabletop_pose.mojo`) on synthetic
frames of the so101-tower's overhead camera.

The frames are drawn by an EXACT ray caster written here: per subsample, the
lens's `unproject` gives the ray and a Cyrus-Beck clip against each prism
gives the hit. The estimator goes the other way (forward projection of edge
samples + convex hull + subsampled coverage), so the two share only the lens
model, which `test_fisheye.mojo` gates against OpenCV. Through the fisheye the
hull is an approximation of the curved silhouette; this gate measures it.

  1. Round trip: `project` then `plane_point` returns the point (both lenses).
  2. Brick (25 mm cube) and bowl (octagon) at placements across the task's
     regions, yaws including 0, 17, 30, 44 and 71 deg — a yaw-blind estimator
     passes a yaw-0 case — through the sim pinhole AND the rig's measured
     fisheye: brick <= 1.5 mm and <= 2 deg, bowl <= 2 mm.
  3. CONTROL, the lens matters: the fisheye frames read with a pinhole of the
     same focal length and centre put the BOWL > 10 mm off somewhere. Without
     it, (2) passing through the fisheye could mean the estimator never used
     the lens. The bowl, not the brick: the optical axis meets the desk at
     (0.28, -0.11), inside the brick's strip, where tan(theta) ~ theta and a
     pinhole reading is only ~1 mm off; the bowl's strip is 25-30 deg off axis.
  4. A dark box over half the brick: `coverage` drops below 0.8 (the caller's
     occlusion signal).
  5. A brick-coloured box OUTSIDE the ROI, bigger than the brick: the estimate
     is still the brick (what keeps the blue tower stand out).

Needs no shim, no camera, no fixture. The camera's pose is the asset's
(`tasks/so101_tower_camera_pose.tower_sim_camera`); the fisheye terms are
`projects/so101-tower/cameras/camera_overhead.txt` of 2026-09-22, copied
below so the gate does not depend on a project checkout.
"""

from std.math import sqrt, cos, sin, pi
from std.sys import exit

from noeira.math3d import Mat3 as Mat3Generic, Vec3 as Vec3Generic
from noeira.tasks.so101_tower_camera_pose import tower_sim_camera
from noeira.vision.fisheye import FisheyeLens, Pinhole
from noeira.vision.tabletop_pose import (
    RigCamera, ColorClass, PrismModel, DeskROI, PoseEstimate,
    estimate_prism_pose,
)

comptime Vec3d = Vec3Generic[DType.float64]
comptime Mat3d = Mat3Generic[DType.float64]

comptime W = 640
comptime H = 480
comptime FOVY = 73.7398
comptime DESK_Z = 0.002
comptime RENDER_SUB = 3

comptime RGB = Tuple[Float64, Float64, Float64]
comptime BRICK_RGB: RGB = (0.17, 0.474, 0.662)
"""`brick_pla` of the calibrated look (26a862992)."""
comptime BOWL_RGB: RGB = (1.0, 0.66, 0.09)
"""`bowl_pla` of the calibrated look (26a862992)."""
comptime DESK_RGB: RGB = (0.93, 0.93, 0.91)
comptime DARK_RGB: RGB = (0.12, 0.12, 0.12)


def check(mut fails: Int, name: String, ok: Bool, detail: String):
    if ok:
        print("  PASS  " + name + "  " + detail)
    else:
        fails += 1
        print("  FAIL  " + name + "  " + detail)


def overhead_lens() -> FisheyeLens:
    return FisheyeLens(
        277.0651676702381, 276.8260365968215, 317.68999154400126,
        257.0689880889907, 0.042650588499554026, -0.03657081959694618,
        0.02219351211116447, -0.007047267919820604, W, H,
    )


# ─── the exact ray caster ──────────────────────────────────────────────────


@fieldwise_init
struct Solid(Copyable, Movable):
    var fx: List[Float64]
    var fy: List[Float64]
    var z0: Float64
    var z1: Float64
    var x: Float64
    var y: Float64
    var yaw: Float64
    var r: Float64
    var g: Float64
    var b: Float64


def solid_of(
    m: PrismModel, x: Float64, y: Float64, yaw: Float64, rgb: RGB,
    z0: Float64 = DESK_Z, lift: Float64 = 0.0,
) -> Solid:
    return Solid(
        m.fx.copy(), m.fy.copy(), z0 + lift, z0 + lift + m.height, x, y, yaw,
        rgb[0], rgb[1], rgb[2],
    )


def hit(s: Solid, o: Vec3d, d: Vec3d) -> Tuple[Float64, Bool]:
    """(t, is_top_face) of the first hit, t = -1 on a miss. Object frame:
    translate then rotate by -yaw; z slab then one half-plane per edge."""
    var c = cos(s.yaw)
    var sn = sin(s.yaw)
    var ox = Float64(o.x) - s.x
    var oy = Float64(o.y) - s.y
    var lx = c * ox + sn * oy
    var ly = -sn * ox + c * oy
    var dx = c * Float64(d.x) + sn * Float64(d.y)
    var dy = -sn * Float64(d.x) + c * Float64(d.y)
    var oz = Float64(o.z)
    var dz = Float64(d.z)
    var t_in = -1.0e30
    var t_out = 1.0e30
    var top = False
    if abs(dz) < 1e-15:
        if oz < s.z0 or oz > s.z1:
            return (-1.0, False)
    else:
        var ta = (s.z0 - oz) / dz
        var tb = (s.z1 - oz) / dz
        var t_near = min(ta, tb)
        t_in = t_near
        t_out = max(ta, tb)
        top = tb < ta  # entering through z1 (looking down)
    var n = len(s.fx)
    for k in range(n):
        var k1 = (k + 1) % n
        var ex = s.fx[k1] - s.fx[k]
        var ey = s.fy[k1] - s.fy[k]
        # outward normal of a CCW polygon
        var nx = ey
        var ny = -ex
        var num = nx * (lx - s.fx[k]) + ny * (ly - s.fy[k])
        var den = nx * dx + ny * dy
        if abs(den) < 1e-15:
            if num > 0.0:
                return (-1.0, False)
            continue
        var t = -num / den
        if den < 0.0:
            if t > t_in:
                t_in = t
                top = False
        else:
            t_out = min(t_out, t)
    if t_in > t_out or t_out <= 0.0 or t_in <= 0.0:
        return (-1.0, False)
    return (t_in, top)


def render(cam: RigCamera, solids: List[Solid]) raises -> List[UInt8]:
    var out = List[UInt8](length=W * H * 3, fill=UInt8(0))
    var o = cam.pos
    for v in range(H):
        for u in range(W):
            var acc_r = 0.0
            var acc_g = 0.0
            var acc_b = 0.0
            for sj in range(RENDER_SUB):
                for si in range(RENDER_SUB):
                    var uu = Float64(u) + (Float64(si) + 0.5) / Float64(RENDER_SUB) - 0.5
                    var vv = Float64(v) + (Float64(sj) + 0.5) / Float64(RENDER_SUB) - 0.5
                    var d = cam.ray(uu, vv)
                    var best_t = 1.0e30
                    var col = DESK_RGB
                    for i in range(len(solids)):
                        var h = hit(solids[i], o, d)
                        if h[0] > 0.0 and h[0] < best_t:
                            best_t = h[0]
                            var shade = 1.0 if h[1] else 0.72
                            col = (
                                solids[i].r * shade, solids[i].g * shade,
                                solids[i].b * shade,
                            )
                    acc_r += col[0]
                    acc_g += col[1]
                    acc_b += col[2]
            var inv = 1.0 / Float64(RENDER_SUB * RENDER_SUB)
            var k = (v * W + u) * 3
            out[k] = UInt8(Int(acc_r * inv * 255.0 + 0.5))
            out[k + 1] = UInt8(Int(acc_g * inv * 255.0 + 0.5))
            out[k + 2] = UInt8(Int(acc_b * inv * 255.0 + 0.5))
    return out^


# ─── helpers ───────────────────────────────────────────────────────────────


def yaw_err_deg(a: Float64, b: Float64, period: Float64) -> Float64:
    var d = a - b
    d = d - period * Float64(Int((d / period) + 1000.5) - 1000)
    return abs(d) * 180.0 / pi


def mm(ax: Float64, ay: Float64, bx: Float64, by: Float64) -> Float64:
    return sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by)) * 1000.0


def brick_class() -> ColorClass:
    return ColorClass.tower_brick_sim()


def bowl_class() -> ColorClass:
    return ColorClass.tower_bowl_sim()


def roi() -> DeskROI:
    return DeskROI(DESK_Z, 0.10, 0.50, -0.25, 0.25)


def main() raises:
    var fails = 0
    var sim = tower_sim_camera("overhead_cam")
    if not sim.found:
        print("FAIL: no overhead_cam in the tower scene")
        exit(1)
    print("overhead_cam at", sim.pos)
    var lens = overhead_lens()
    var pin_cam = RigCamera(Pinhole.sim(FOVY, W, H), sim.pos, sim.rot)
    var fish_cam = RigCamera(lens, sim.pos, sim.rot)
    var naive_cam = RigCamera(
        Pinhole(lens.fx, lens.fy, lens.cx, lens.cy, W, H), sim.pos, sim.rot
    )
    var brick = PrismModel.tower_brick()
    var bowl = PrismModel.tower_bowl()

    # 1. round trip
    print("\n1. project -> plane_point round trip")
    for ci in range(2):
        var worst = 0.0
        var rc = pin_cam.copy() if ci == 0 else fish_cam.copy()
        for k in range(25):
            var x = 0.12 + 0.3 * Float64(k % 5) / 4.0
            var y = -0.2 + 0.4 * Float64(k // 5) / 4.0
            var z = DESK_Z + 0.01 * Float64(k % 3)
            var uv = rc.project(Vec3d(x, y, z))
            var p = rc.plane_point(uv[0], uv[1], z)
            worst = max(worst, mm(p[0], p[1], x, y))
        check(
            fails, "round trip " + ("pinhole" if ci == 0 else "fisheye"),
            worst < 1e-6, "worst " + String(worst) + " mm",
        )

    # 2 + 3. placements
    # (brick x, y, yaw deg, bowl x, y, yaw deg) — the task's strips:
    # desk_right y in [-0.16, -0.06], desk_left y in [0.06, 0.16], x in [0.18, 0.38]
    var cases: List[List[Float64]] = [
        [0.28, -0.11, 0.0, 0.28, 0.11, 0.0],
        [0.20, -0.07, 17.0, 0.20, 0.07, 10.0],
        [0.36, -0.15, 30.0, 0.36, 0.15, 22.0],
        [0.24, -0.14, 44.0, 0.33, 0.08, 31.0],
        [0.33, -0.08, 71.0, 0.22, 0.14, 40.0],
        [0.19, -0.16, 58.0, 0.37, 0.10, 5.0],
    ]
    var naive_worst = 0.0
    var naive_brick = 0.0
    for ci in range(2):
        var cname = String("pinhole") if ci == 0 else String("fisheye")
        print("\n2. placements through the " + cname)
        var bw_xy = 0.0
        var bw_yaw = 0.0
        var bw0 = 0.0
        var ow_xy = 0.0
        for c in cases:
            var solids = List[Solid]()
            solids.append(solid_of(brick, c[0], c[1], c[2] * pi / 180.0, BRICK_RGB))
            solids.append(solid_of(bowl, c[3], c[4], c[5] * pi / 180.0, BOWL_RGB))
            var cam = pin_cam.copy() if ci == 0 else fish_cam.copy()
            var frame = render(cam, solids)
            var eb = estimate_prism_pose(frame, cam, brick_class(), brick, roi())
            var eo = estimate_prism_pose(frame, cam, bowl_class(), bowl, roi())
            if not eb.found or not eo.found:
                check(fails, cname + " found", False, String(eb) + " | " + String(eo))
                continue
            var exy = mm(eb.x, eb.y, c[0], c[1])
            var eyaw = yaw_err_deg(eb.yaw, c[2] * pi / 180.0, brick.period)
            var e0 = mm(eb.x0, eb.y0, c[0], c[1])
            var oxy = mm(eo.x, eo.y, c[3], c[4])
            print(
                "   brick (", c[0], c[1], c[2], "deg ) err", exy, "mm", eyaw,
                "deg (start", e0, "mm), px", eb.n_px, "cov", eb.coverage,
                "| bowl err", oxy, "mm, px", eo.n_px,
            )
            bw_xy = max(bw_xy, exy)
            bw_yaw = max(bw_yaw, eyaw)
            bw0 = max(bw0, e0)
            ow_xy = max(ow_xy, oxy)
            if ci == 1:
                var en = estimate_prism_pose(frame, naive_cam, bowl_class(), bowl, roi())
                if en.found:
                    naive_worst = max(naive_worst, mm(en.x, en.y, c[3], c[4]))
                var enb = estimate_prism_pose(frame, naive_cam, brick_class(), brick, roi())
                if enb.found:
                    naive_brick = max(naive_brick, mm(enb.x, enb.y, c[0], c[1]))
        check(fails, cname + " brick xy", bw_xy <= 1.5, "worst " + String(bw_xy) + " mm (centroid start " + String(bw0) + ")")
        check(fails, cname + " brick yaw", bw_yaw <= 2.0, "worst " + String(bw_yaw) + " deg")
        check(fails, cname + " bowl xy", ow_xy <= 2.0, "worst " + String(ow_xy) + " mm")
    print("\n3. control: the fisheye frames read as a pinhole")
    check(
        fails, "lens matters (bowl)", naive_worst > 10.0,
        "worst " + String(naive_worst) + " mm (must be > 10); brick, near the axis, "
        + String(naive_brick) + " mm",
    )

    # 4. occlusion: a dark box 3 cm above the brick. The camera looks 25 deg
    # toward +x, so a box at height ~4 cm lands ~19 mm further along +x in the
    # image than its x: centred 34 mm short of the brick, it hides the -x half.
    print("\n4. a dark box over half the brick")
    var occl = List[Solid]()
    occl.append(solid_of(brick, 0.28, -0.11, 0.3, BRICK_RGB))
    var lid = PrismModel.square("lid", 0.03, 0.02)
    occl.append(solid_of(lid, 0.28 - 0.034, -0.11, 0.0, DARK_RGB, lift=0.03))
    var fo = render(fish_cam, occl)
    var eoc = estimate_prism_pose(fo, fish_cam, brick_class(), brick, roi())
    print("   ", eoc)
    check(fails, "occlusion lowers coverage", eoc.found and eoc.coverage < 0.8, "coverage " + String(eoc.coverage))

    # 5. a brick-coloured box outside the ROI, larger than the brick
    print("\n5. a brick-coloured distractor outside the ROI")
    var dis = List[Solid]()
    dis.append(solid_of(brick, 0.30, -0.12, 0.5, BRICK_RGB))
    var slab = PrismModel.square("slab", 0.05, 0.03)
    dis.append(solid_of(slab, 0.06, -0.12, 0.0, BRICK_RGB))
    var fd = render(fish_cam, dis)
    var ed = estimate_prism_pose(fd, fish_cam, brick_class(), brick, roi())
    var edx = mm(ed.x, ed.y, 0.30, -0.12)
    print("   ", ed)
    check(fails, "ROI keeps the distractor out", ed.found and edx <= 1.5, "err " + String(edx) + " mm")

    print()
    if fails > 0:
        print("FAILED:", fails)
        exit(1)
    print("ALL PASS")
