# +--------------------------------------------------------------------------+ #
# | Tabletop object poses from one calibrated camera
# +--------------------------------------------------------------------------+ #
"""The pose (x, y, yaw) of a known prism resting on a known plane, from one
RGB frame of a calibrated camera.

    from noeira.vision.tabletop_pose import (
        RigCamera, ColorClass, PrismModel, DeskROI, estimate_prism_pose,
    )
    var cam = RigCamera(lens, cam_pos_world, cam_rot_mujoco)   # or a Pinhole
    var est = estimate_prism_pose(
        frame_rgb_hwc, cam, ColorClass(205.0, 18.0, 0.45, 0.12, 1.0),
        PrismModel.square("brick", 0.025, 0.025), DeskROI(...),
    )

WHAT IT IS FOR. The object-pose track (`OBJECT_POSE_POLICY_PLAN.md` in the
private docs) puts the sim-to-real boundary at object POSES: a policy reads
the joints plus the cube's and the bowl's pose on the desk, and on the rig
this module produces those poses from the overhead camera. Pure Mojo, host
float64, no OpenCV and no GPU: it runs in the deploy process on the Jetson.

## The method: segment, then fit the known shape

1. A colour mask (HSV box, `ColorClass`), kept only where the pixel's ray
   meets the desk plane inside a world-frame ROI (`DeskROI`). The ROI is what
   keeps the blue tower stand — the brick's hue in the overhead view — and
   everything off the desk out of the mask.
2. Connected components (4-connected); the object's is the one whose area is
   closest, in log ratio, to the area the model projects to where the
   component back-projects, among those within [`min_coverage`,
   `max_coverage`] of it (0.15..2: occluded yes, a bigger blob no). Holes are filled (the cube in the bowl is a hole
   in the yellow).
3. Analysis by synthesis: the object is a convex PRISM (a footprint polygon
   extruded from the desk up to its height). For a candidate (x, y, yaw) its
   footprint and top outlines and vertical edges are sampled, projected
   through the lens, and their convex hull is rasterised with S x S
   subsamples; the cost is sum |coverage - mask| over a window around the
   component, plus the hull's area that falls outside the window. Coarse yaw
   over the symmetry period with a centroid alignment at each yaw, then a
   pattern search on (x, y, yaw).

⚠ The hull of projected points IS the silhouette for a pinhole. Through a
fisheye a straight edge is a curve; the edges are sampled (`EDGE_SAMPLES`)
so the hull follows it. `tests/vision/test_tabletop_pose.mojo` draws its
frames with an EXACT ray caster that shares only the lens with this file, so
this approximation is inside what that gate measures.

## Conventions

- Frames are RGB, HWC, uint8, row 0 at the top. Pixel coordinates are
  OpenCV's (integer = pixel centre), as `vision/fisheye.mojo`.
- The camera pose is given in MuJoCo axes (x right, y UP, looking down -z) —
  what `tasks/so101_tower_camera_pose.tower_sim_camera` returns — and stored
  in OpenCV axes (y down, +z forward): `R_cv = R_mj @ diag(1, -1, -1)`.
- The pose returned is the footprint's origin (x, y) in the world frame and
  its yaw about world z, reduced to [0, period) — a cube's yaw is only
  defined modulo 90 degrees, an octagon's modulo 45.

## Confidence

`coverage` = component area / the model's area at the fitted pose: < 1 when
something (the gripper) hides part of the object. `residual` = cost / model
area: large when the blob is not the shape. The caller decides what to trust;
a partly hidden object pulls the fit toward its visible part.
"""

from std.math import atan2, cos, sin, sqrt, pi, log, floor, ceil

from noeira.math3d import Mat3 as Mat3Generic, Vec3 as Vec3Generic

from .fisheye import FisheyeLens, Pinhole

comptime Vec3d = Vec3Generic[DType.float64]
comptime Mat3d = Mat3Generic[DType.float64]
comptime P2 = SIMD[DType.float64, 2]

comptime EDGE_SAMPLES = 4
"""Points per prism edge fed to the hull (the fisheye bends edges)."""
comptime SUBSAMPLES = 6
"""S: an S x S grid per pixel when rasterising the model's silhouette."""
comptime MIN_COMPONENT_PX = 6
comptime BIG = 1.0e30


# ─── the camera ────────────────────────────────────────────────────────────


struct RigCamera(Copyable, Movable):
    """A calibrated camera: lens (fisheye or ideal pinhole) + world pose."""

    var is_fisheye: Bool
    var lens: FisheyeLens
    var pin: Pinhole
    var pos: Vec3d
    var rot_cv: Mat3d
    """world <- camera, OpenCV camera axes."""
    var rot_cv_t: Mat3d
    """camera <- world."""
    var width: Int
    var height: Int

    def __init__(out self, lens: FisheyeLens, pos: Vec3d, rot_mj: Mat3d):
        self.is_fisheye = True
        self.lens = lens
        self.pin = Pinhole(lens.fx, lens.fy, lens.cx, lens.cy, lens.width, lens.height)
        self.pos = pos
        self.rot_cv = Mat3d.from_cols(rot_mj.col(0), -rot_mj.col(1), -rot_mj.col(2))
        self.rot_cv_t = self.rot_cv.transpose()
        self.width = lens.width
        self.height = lens.height

    def __init__(out self, pin: Pinhole, pos: Vec3d, rot_mj: Mat3d):
        self.is_fisheye = False
        self.lens = FisheyeLens(
            pin.fx, pin.fy, pin.cx, pin.cy, 0.0, 0.0, 0.0, 0.0, pin.width,
            pin.height,
        )
        self.pin = pin
        self.pos = pos
        self.rot_cv = Mat3d.from_cols(rot_mj.col(0), -rot_mj.col(1), -rot_mj.col(2))
        self.rot_cv_t = self.rot_cv.transpose()
        self.width = pin.width
        self.height = pin.height

    def project(self, p: Vec3d) -> Tuple[Float64, Float64, Bool]:
        """The pixel of world point `p`; False behind the camera."""
        var c = self.rot_cv_t * (p - self.pos)
        if Float64(c.z) <= 1e-9:
            return (0.0, 0.0, False)
        var a = Float64(c.x) / Float64(c.z)
        var b = Float64(c.y) / Float64(c.z)
        if self.is_fisheye:
            var uv = self.lens.project(a, b)
            return (uv[0], uv[1], True)
        return (self.pin.fx * a + self.pin.cx, self.pin.fy * b + self.pin.cy, True)

    def ray(self, u: Float64, v: Float64) raises -> Vec3d:
        """The world direction (not unit) pixel `(u, v)` sees."""
        var a: Float64
        var b: Float64
        if self.is_fisheye:
            var ab = self.lens.unproject(u, v)
            a = ab[0]
            b = ab[1]
        else:
            a = (u - self.pin.cx) / self.pin.fx
            b = (v - self.pin.cy) / self.pin.fy
        return self.rot_cv * Vec3d(a, b, 1.0)

    def plane_point(self, u: Float64, v: Float64, z: Float64) -> Tuple[Float64, Float64, Bool]:
        """Where pixel `(u, v)`'s ray meets the plane at height `z`."""
        try:
            var d = self.ray(u, v)
            if abs(Float64(d.z)) < 1e-12:
                return (0.0, 0.0, False)
            var t = (z - Float64(self.pos.z)) / Float64(d.z)
            if t <= 0.0:
                return (0.0, 0.0, False)
            return (
                Float64(self.pos.x) + t * Float64(d.x),
                Float64(self.pos.y) + t * Float64(d.y),
                True,
            )
        except:
            return (0.0, 0.0, False)


# ─── what to look for ──────────────────────────────────────────────────────


@fieldwise_init
struct ColorClass(Copyable, ImplicitlyCopyable, Movable, Writable):
    """An HSV box: hue within `hue_tol_deg` of `hue_deg` (wrapping), saturation
    >= `s_min`, value in [`v_min`, `v_max`] (all in [0, 1])."""

    var hue_deg: Float64
    var hue_tol_deg: Float64
    var s_min: Float64
    var v_min: Float64
    var v_max: Float64

    def matches(self, r: UInt8, g: UInt8, b: UInt8) -> Bool:
        var hsv = rgb_to_hsv(r, g, b)
        if hsv[1] < self.s_min or hsv[2] < self.v_min or hsv[2] > self.v_max:
            return False
        var dh = abs(hsv[0] - self.hue_deg)
        if dh > 180.0:
            dh = 360.0 - dh
        return dh <= self.hue_tol_deg

    @staticmethod
    def tower_brick_sim() -> Self:
        """The sim brick under the calibrated look (26a862992): `brick_pla`
        (0.17, 0.474, 0.662) is hue 203; the tracer renders it hue 217 in the
        overhead camera and 203 in the wrist camera, s 0.68-0.76, v 0.51-0.56
        (noeira-26's medians). The real brick is 195..216. ⚠ The stand's
        `stand_pla` renders at the SAME hue (216.9): the hue band cannot
        separate them — the ROI does."""
        return Self(210.0, 14.0, 0.35, 0.15, 1.0)

    @staticmethod
    def tower_bowl_sim() -> Self:
        """The sim bowl under the calibrated look (26a862992): `bowl_pla`
        (1.0, 0.66, 0.09) is hue 37.6, rendered hue 38 (s 0.88, v 0.82). A lit
        face that clips red first moves toward yellow (the pre-calibration
        material clipped to hue 60), so the band runs 24..52 — which also
        holds the real printed bowl (29..42 on `cube-in-bowl-printed`)."""
        return Self(38.0, 14.0, 0.45, 0.20, 1.0)

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "hue ", self.hue_deg, "+-", self.hue_tol_deg, " s>=", self.s_min,
            " v in [", self.v_min, ", ", self.v_max, "]",
        )


def rgb_to_hsv(r: UInt8, g: UInt8, b: UInt8) -> Tuple[Float64, Float64, Float64]:
    """(hue deg [0, 360), saturation [0, 1], value [0, 1])."""
    var rf = Float64(r) / 255.0
    var gf = Float64(g) / 255.0
    var bf = Float64(b) / 255.0
    var mx = max(rf, max(gf, bf))
    var mn = min(rf, min(gf, bf))
    var d = mx - mn
    var h = 0.0
    if d > 1e-12:
        if mx == rf:
            h = 60.0 * ((gf - bf) / d)
        elif mx == gf:
            h = 60.0 * ((bf - rf) / d + 2.0)
        else:
            h = 60.0 * ((rf - gf) / d + 4.0)
        if h < 0.0:
            h += 360.0
    var s = d / mx if mx > 1e-12 else 0.0
    return (h, s, mx)


struct PrismModel(Copyable, Movable, Writable):
    """A convex footprint polygon (object frame, CCW, metres) extruded from
    the desk plane up to `height`. The pose is the footprint origin."""

    var name: String
    var fx: List[Float64]
    var fy: List[Float64]
    var height: Float64
    var period: Float64
    """Yaw symmetry: the pose's yaw is reported in [0, period)."""

    def __init__(
        out self, name: String, var fx: List[Float64], var fy: List[Float64],
        height: Float64, period: Float64,
    ):
        self.name = name
        self.fx = fx^
        self.fy = fy^
        self.height = height
        self.period = period

    @staticmethod
    def regular(
        name: String, n: Int, circumradius: Float64, height: Float64
    ) -> Self:
        """A regular n-gon whose FACES are normal to x at yaw 0 (vertices at
        pi/n + 2 pi k / n) — the convention of both tower props' MJCF."""
        var fx = List[Float64]()
        var fy = List[Float64]()
        for k in range(n):
            var a = pi / Float64(n) + 2.0 * pi * Float64(k) / Float64(n)
            fx.append(circumradius * cos(a))
            fy.append(circumradius * sin(a))
        return Self(name, fx^, fy^, height, 2.0 * pi / Float64(n))

    @staticmethod
    def square(name: String, side: Float64, height: Float64) -> Self:
        return Self.regular(name, 4, side / sqrt(2.0), height)

    @staticmethod
    def tower_brick() -> Self:
        """The printed 25 mm cube (`tasks/assets/props/brick.xml`, box
        half-size 0.0125, faces normal to x)."""
        return Self.square("brick", 0.025, 0.025)

    @staticmethod
    def tower_bowl() -> Self:
        """The printed octagonal bowl (`tasks/assets/props/bowl.xml`: outer
        circumradius 0.060614, 45 mm tall, wall0 normal to x)."""
        return Self.regular("bowl", 8, 0.060614, 0.045)

    def n(self) -> Int:
        return len(self.fx)

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            self.name, " (", self.n(), "-gon, h ", self.height * 1000.0,
            " mm, period ", self.period * 180.0 / pi, " deg)",
        )


@fieldwise_init
struct DeskROI(Copyable, ImplicitlyCopyable, Movable, Writable):
    """The desk plane's height and the world (x, y) box objects may be in."""

    var z: Float64
    var x_min: Float64
    var x_max: Float64
    var y_min: Float64
    var y_max: Float64

    def inside(self, x: Float64, y: Float64) -> Bool:
        return x >= self.x_min and x <= self.x_max and y >= self.y_min and y <= self.y_max

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "desk z ", self.z, " x [", self.x_min, ", ", self.x_max, "] y [",
            self.y_min, ", ", self.y_max, "]",
        )


@fieldwise_init
struct PoseEstimate(Copyable, ImplicitlyCopyable, Movable, Writable):
    var found: Bool
    var x: Float64
    var y: Float64
    var yaw: Float64
    """In [0, model period)."""
    var x0: Float64
    """The centroid back-projection before the fit (the fit's start)."""
    var y0: Float64
    var n_px: Int
    """The component's pixels (holes filled)."""
    var coverage: Float64
    var residual: Float64
    var n_components: Int
    """Components of the colour that were in the ROI."""

    @staticmethod
    def none(n_components: Int) -> Self:
        return Self(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, n_components)

    def write_to(self, mut writer: Some[Writer]):
        if not self.found:
            writer.write("not found (", self.n_components, " components)")
            return
        writer.write(
            "x ", self.x, " y ", self.y, " yaw ", self.yaw * 180.0 / pi,
            " deg, ", self.n_px, " px, coverage ", self.coverage,
            " residual ", self.residual,
        )


# ─── the silhouette ────────────────────────────────────────────────────────


def _cross(o: P2, a: P2, b: P2) -> Float64:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(var pts: List[P2]) -> List[P2]:
    """Andrew's monotone chain; CCW in pixel coordinates (v down), no
    repeated end point."""
    var n = len(pts)
    # insertion sort by (u, v): n is ~100
    for i in range(1, n):
        var p = pts[i]
        var j = i - 1
        while j >= 0 and (pts[j][0] > p[0] or (pts[j][0] == p[0] and pts[j][1] > p[1])):
            pts[j + 1] = pts[j]
            j -= 1
        pts[j + 1] = p
    if n < 3:
        return pts^
    var hull = List[P2]()
    for i in range(n):
        while len(hull) >= 2 and _cross(hull[len(hull) - 2], hull[len(hull) - 1], pts[i]) <= 0.0:
            _ = hull.pop()
        hull.append(pts[i])
    var lower = len(hull) + 1
    var i = n - 2
    while i >= 0:
        while len(hull) >= lower and _cross(hull[len(hull) - 2], hull[len(hull) - 1], pts[i]) <= 0.0:
            _ = hull.pop()
        hull.append(pts[i])
        i -= 1
    _ = hull.pop()
    return hull^


def model_silhouette(
    cam: RigCamera, model: PrismModel, desk_z: Float64, x: Float64, y: Float64,
    yaw: Float64,
) -> List[P2]:
    """The convex hull of the prism's projected outline; empty if any
    sample is behind the camera."""
    var c = cos(yaw)
    var s = sin(yaw)
    var n = model.n()
    var pts = List[P2]()
    var zs = (desk_z, desk_z + model.height)
    for k in range(n):
        var ax = x + c * model.fx[k] - s * model.fy[k]
        var ay = y + s * model.fx[k] + c * model.fy[k]
        var k1 = (k + 1) % n
        var bx = x + c * model.fx[k1] - s * model.fy[k1]
        var by = y + s * model.fx[k1] + c * model.fy[k1]
        for level in range(2):
            var z = zs[0] if level == 0 else zs[1]
            for j in range(EDGE_SAMPLES):
                var t = Float64(j) / Float64(EDGE_SAMPLES)
                var uv = cam.project(Vec3d(ax + t * (bx - ax), ay + t * (by - ay), z))
                if not uv[2]:
                    return List[P2]()
                pts.append(P2(uv[0], uv[1]))
        # the vertical edge at vertex k (its ends are in the loops above)
        for j in range(1, EDGE_SAMPLES):
            var t = Float64(j) / Float64(EDGE_SAMPLES)
            var uv = cam.project(Vec3d(ax, ay, zs[0] + t * (zs[1] - zs[0])))
            if not uv[2]:
                return List[P2]()
            pts.append(P2(uv[0], uv[1]))
    return convex_hull(pts^)


def polygon_area(poly: List[P2]) -> Float64:
    var a = 0.0
    var n = len(poly)
    for i in range(n):
        var p = poly[i]
        var q = poly[(i + 1) % n]
        a += p[0] * q[1] - q[0] * p[1]
    return abs(a) * 0.5


def _row_span(poly: List[P2], v: Float64) -> Tuple[Float64, Float64, Bool]:
    """The u interval where the horizontal line at `v` is inside the convex
    polygon."""
    var lo = BIG
    var hi = -BIG
    var n = len(poly)
    for i in range(n):
        var p = poly[i]
        var q = poly[(i + 1) % n]
        var v0 = p[1]
        var v1 = q[1]
        if (v0 <= v and v1 >= v) or (v1 <= v and v0 >= v):
            var u: Float64
            if abs(v1 - v0) < 1e-12:
                lo = min(lo, min(p[0], q[0]))
                hi = max(hi, max(p[0], q[0]))
                continue
            u = p[0] + (v - v0) / (v1 - v0) * (q[0] - p[0])
            lo = min(lo, u)
            hi = max(hi, u)
    return (lo, hi, lo <= hi)


@fieldwise_init
struct _Eval(Copyable, ImplicitlyCopyable, Movable):
    var cost: Float64
    var area: Float64
    """The silhouette's area in pixels (subsample count, whole image)."""
    var cu: Float64
    """Coverage-weighted centroid of the silhouette."""
    var cv: Float64


struct _Window(Movable):
    """The component's mask on a window of the image."""

    var u0: Int
    var v0: Int
    var w: Int
    var h: Int
    var m: List[Float64]
    var area: Float64
    var cu: Float64
    var cv: Float64

    def __init__(out self, u0: Int, v0: Int, w: Int, h: Int):
        self.u0 = u0
        self.v0 = v0
        self.w = w
        self.h = h
        self.m = List[Float64](length=w * h, fill=0.0)
        self.area = 0.0
        self.cu = 0.0
        self.cv = 0.0

    def finish(mut self):
        var a = 0.0
        var su = 0.0
        var sv = 0.0
        for j in range(self.h):
            for i in range(self.w):
                var x = self.m[j * self.w + i]
                a += x
                su += x * Float64(self.u0 + i)
                sv += x * Float64(self.v0 + j)
        self.area = a
        self.cu = su / a if a > 0.0 else 0.0
        self.cv = sv / a if a > 0.0 else 0.0


def _evaluate(
    cam: RigCamera, model: PrismModel, desk_z: Float64, x: Float64, y: Float64,
    yaw: Float64, win: _Window,
) -> _Eval:
    var poly = model_silhouette(cam, model, desk_z, x, y, yaw)
    if len(poly) < 3:
        return _Eval(BIG, 0.0, 0.0, 0.0)
    var total_area = polygon_area(poly)
    var s = SUBSAMPLES
    var inv = 1.0 / Float64(s * s)
    var cost = 0.0
    var in_area = 0.0
    var su = 0.0
    var sv = 0.0
    var cov_row = List[Float64](length=win.w, fill=0.0)
    for j in range(win.h):
        var pv = Float64(win.v0 + j)
        for i in range(win.w):
            cov_row[i] = 0.0
        for sj in range(s):
            var v = pv + (Float64(sj) + 0.5) / Float64(s) - 0.5
            var span = _row_span(poly, v)
            if not span[2]:
                continue
            # subsample columns j' of pixel u lie at u + (j' + 0.5)/s - 0.5
            var i_lo = max(0, Int(floor(span[0] - Float64(win.u0))) - 1)
            var i_hi = min(win.w - 1, Int(ceil(span[1] - Float64(win.u0))) + 1)
            for i in range(i_lo, i_hi + 1):
                var pu = Float64(win.u0 + i)
                var a = (span[0] - pu + 0.5) * Float64(s) - 0.5
                var b = (span[1] - pu + 0.5) * Float64(s) - 0.5
                var k0 = max(0, Int(ceil(a)))
                var k1 = min(s - 1, Int(floor(b)))
                if k1 >= k0:
                    cov_row[i] += Float64(k1 - k0 + 1) * inv
        for i in range(win.w):
            var c = cov_row[i]
            cost += abs(c - win.m[j * win.w + i])
            in_area += c
            su += c * Float64(win.u0 + i)
            sv += c * Float64(win.v0 + j)
    # silhouette area outside the window is unmatched by construction (only
    # added when the hull leaves it: the subsampled area differs from the
    # exact one by a quantisation jitter the search would otherwise see)
    var inside = True
    for p in poly:
        if (
            p[0] < Float64(win.u0) - 0.5 or p[0] > Float64(win.u0 + win.w) - 0.5
            or p[1] < Float64(win.v0) - 0.5 or p[1] > Float64(win.v0 + win.h) - 0.5
        ):
            inside = False
            break
    if not inside:
        cost += max(0.0, total_area - in_area)
    var cu = su / in_area if in_area > 0.0 else 0.0
    var cv = sv / in_area if in_area > 0.0 else 0.0
    return _Eval(cost, total_area, cu, cv)


def expected_area_px(
    cam: RigCamera, model: PrismModel, desk_z: Float64, x: Float64, y: Float64,
    yaw: Float64,
) -> Float64:
    var poly = model_silhouette(cam, model, desk_z, x, y, yaw)
    if len(poly) < 3:
        return 0.0
    return polygon_area(poly)


# ─── the estimator ─────────────────────────────────────────────────────────


def _wrap(yaw: Float64, period: Float64) -> Float64:
    var y = yaw - period * floor(yaw / period)
    if y >= period:
        y -= period
    return y


def estimate_prism_pose(
    frame: List[UInt8], cam: RigCamera, color: ColorClass, model: PrismModel,
    roi: DeskROI, min_coverage: Float64 = 0.15, max_coverage: Float64 = 2.0,
) raises -> PoseEstimate:
    """The pose of `model` in `frame` (RGB HWC uint8 at the camera's size);
    see the module header."""
    var w = cam.width
    var h = cam.height
    if len(frame) != w * h * 3:
        raise Error(
            "estimate_prism_pose: frame has " + String(len(frame))
            + " bytes, the camera is " + String(w) + "x" + String(h) + "x3"
        )
    var mid_z = roi.z + 0.5 * model.height

    # 1. the colour mask, in the ROI
    var mask = List[UInt8](length=w * h, fill=UInt8(0))
    for v in range(h):
        for u in range(w):
            var k = (v * w + u) * 3
            if not color.matches(frame[k], frame[k + 1], frame[k + 2]):
                continue
            var p = cam.plane_point(Float64(u), Float64(v), mid_z)
            if p[2] and roi.inside(p[0], p[1]):
                mask[v * w + u] = 1

    # 2. connected components; keep the best-sized one
    var label = List[Int32](length=w * h, fill=Int32(-1))
    var stack = List[Int]()
    var best_score = BIG
    var best_label = -1
    var best_bbox = (0, 0, 0, 0)
    var best_xy = (0.0, 0.0)
    var n_comp = 0
    for start in range(w * h):
        if mask[start] == 0 or label[start] >= 0:
            continue
        var lab = Int32(n_comp)
        n_comp += 1
        label[start] = lab
        stack.clear()
        stack.append(start)
        var cnt = 0
        var sx = 0.0
        var sy = 0.0
        var nxy = 0
        var umin = w
        var umax = -1
        var vmin = h
        var vmax = -1
        while len(stack) > 0:
            var idx = stack.pop()
            var u = idx % w
            var v = idx // w
            cnt += 1
            umin = min(umin, u)
            umax = max(umax, u)
            vmin = min(vmin, v)
            vmax = max(vmax, v)
            var p = cam.plane_point(Float64(u), Float64(v), mid_z)
            if p[2]:
                sx += p[0]
                sy += p[1]
                nxy += 1
            if u > 0 and mask[idx - 1] != 0 and label[idx - 1] < 0:
                label[idx - 1] = lab
                stack.append(idx - 1)
            if u < w - 1 and mask[idx + 1] != 0 and label[idx + 1] < 0:
                label[idx + 1] = lab
                stack.append(idx + 1)
            if v > 0 and mask[idx - w] != 0 and label[idx - w] < 0:
                label[idx - w] = lab
                stack.append(idx - w)
            if v < h - 1 and mask[idx + w] != 0 and label[idx + w] < 0:
                label[idx + w] = lab
                stack.append(idx + w)
        if cnt < MIN_COMPONENT_PX or nxy == 0:
            continue
        var cx = sx / Float64(nxy)
        var cy = sy / Float64(nxy)
        var expect = expected_area_px(cam, model, roi.z, cx, cy, 0.0)
        if expect <= 0.0:
            continue
        var ratio = Float64(cnt) / expect
        # a blob more than twice the model's own projected area is not the
        # object — on the rig, blue floor seen past the desk's edge, whose
        # rays meet the desk plane inside the ROI, beat a brick half hidden
        # by the bowl's wall on |log ratio| alone
        if ratio < min_coverage or ratio > max_coverage:
            continue
        var score = abs(log(ratio))
        if score < best_score:
            best_score = score
            best_label = Int(lab)
            best_bbox = (umin, umax, vmin, vmax)
            best_xy = (cx, cy)
    if best_label < 0:
        return PoseEstimate.none(n_comp)

    # 3. the window: the component's box plus a margin of half its size
    var bw = best_bbox[1] - best_bbox[0] + 1
    var bh = best_bbox[3] - best_bbox[2] + 1
    var margin = max(6, max(bw, bh) // 2)
    var u0 = max(0, best_bbox[0] - margin)
    var v0 = max(0, best_bbox[2] - margin)
    var u1 = min(w - 1, best_bbox[1] + margin)
    var v1 = min(h - 1, best_bbox[3] + margin)
    var win = _Window(u0, v0, u1 - u0 + 1, v1 - v0 + 1)
    for j in range(win.h):
        for i in range(win.w):
            if Int(label[(v0 + j) * w + (u0 + i)]) == best_label:
                win.m[j * win.w + i] = 1.0
    _fill_holes(win)
    win.finish()

    # 4. fit: coarse yaw with centroid alignment, then a pattern search
    var x0 = best_xy[0]
    var y0 = best_xy[1]
    var jac = _plane_jacobian(cam, win.cu, win.cv, mid_z)
    var n_yaw = 12
    var best = _Eval(BIG, 0.0, 0.0, 0.0)
    var bx = x0
    var by = y0
    var byaw = 0.0
    for k in range(n_yaw):
        var yaw = model.period * Float64(k) / Float64(n_yaw)
        var x = x0
        var y = y0
        for _ in range(3):
            var ea = _evaluate(cam, model, roi.z, x, y, yaw, win)
            if ea.cost >= BIG:
                break
            var du = win.cu - ea.cu
            var dv = win.cv - ea.cv
            x += jac[0] * du + jac[1] * dv
            y += jac[2] * du + jac[3] * dv
        var e = _evaluate(cam, model, roi.z, x, y, yaw, win)
        if e.cost < best.cost:
            best = e
            bx = x
            by = y
            byaw = yaw
    var step_xy = 0.002
    var step_yaw = model.period / Float64(n_yaw) / 2.0
    var iters = 0
    while step_xy > 1.0e-4 and iters < 400:
        iters += 1
        var improved = False
        for mv in range(6):
            var x = bx
            var y = by
            var yaw = byaw
            if mv == 0:
                x += step_xy
            elif mv == 1:
                x -= step_xy
            elif mv == 2:
                y += step_xy
            elif mv == 3:
                y -= step_xy
            elif mv == 4:
                yaw += step_yaw
            else:
                yaw -= step_yaw
            var e = _evaluate(cam, model, roi.z, x, y, yaw, win)
            if e.cost < best.cost:
                best = e
                bx = x
                by = y
                byaw = yaw
                improved = True
        if not improved:
            step_xy *= 0.5
            step_yaw *= 0.5
    var coverage = win.area / best.area if best.area > 0.0 else 0.0
    var residual = best.cost / best.area if best.area > 0.0 else BIG
    return PoseEstimate(
        True, bx, by, _wrap(byaw, model.period), x0, y0, Int(win.area),
        coverage, residual, n_comp,
    )


def _plane_jacobian(
    cam: RigCamera, u: Float64, v: Float64, z: Float64
) -> Tuple[Float64, Float64, Float64, Float64]:
    """d(x, y)/d(u, v) of the ray-plane map at a pixel: (dx/du, dx/dv, dy/du,
    dy/dv)."""
    var p = cam.plane_point(u, v, z)
    var pu = cam.plane_point(u + 1.0, v, z)
    var pv = cam.plane_point(u, v + 1.0, z)
    if not (p[2] and pu[2] and pv[2]):
        return (0.0, 0.0, 0.0, 0.0)
    return (pu[0] - p[0], pv[0] - p[0], pu[1] - p[1], pv[1] - p[1])


def _fill_holes(mut win: _Window):
    """Set every window pixel not reachable from the window's border through
    non-component pixels."""
    var n = win.w * win.h
    var outside = List[UInt8](length=n, fill=UInt8(0))
    var stack = List[Int]()
    for j in range(win.h):
        for i in range(win.w):
            if not (i == 0 or j == 0 or i == win.w - 1 or j == win.h - 1):
                continue
            var k = j * win.w + i
            if win.m[k] == 0.0 and outside[k] == 0:
                outside[k] = 1
                stack.append(k)
    while len(stack) > 0:
        var k = stack.pop()
        var i = k % win.w
        var j = k // win.w
        if i > 0 and win.m[k - 1] == 0.0 and outside[k - 1] == 0:
            outside[k - 1] = 1
            stack.append(k - 1)
        if i < win.w - 1 and win.m[k + 1] == 0.0 and outside[k + 1] == 0:
            outside[k + 1] = 1
            stack.append(k + 1)
        if j > 0 and win.m[k - win.w] == 0.0 and outside[k - win.w] == 0:
            outside[k - win.w] = 1
            stack.append(k - win.w)
        if j < win.h - 1 and win.m[k + win.w] == 0.0 and outside[k + win.w] == 0:
            outside[k + win.w] = 1
            stack.append(k + win.w)
    for k in range(n):
        if outside[k] == 0:
            win.m[k] = 1.0
