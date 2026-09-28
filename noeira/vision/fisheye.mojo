# +--------------------------------------------------------------------------+ #
# | Fisheye frame -> the sim's pinhole camera
# +--------------------------------------------------------------------------+ #
"""Undistort a real fisheye frame to the ideal pinhole the simulator renders.

`docs/camera-rig.md` §6: the rig's two cameras are a 170-degree module, MuJoCo
cameras are ideal pinholes, and the sim's `overhead_cam` / `wrist_cam` are a
73.74-degree `fovy` at 640x480. A policy trained on the tracer's frames and
fed raw fisheye frames sees a different lens, and no domain randomization
closes that gap. So the REAL side is brought to the sim's camera model:

    fisheye frame (640x480)  --UndistortMap-->  pinhole frame (640x480, fovy
    73.74, the tracer's pixel convention)  --PIL-exact resize-->  320x240

One implementation, used by the dataset importer (`data/lerobot.mojo`,
`--undistort`) and by the real-robot deploy — the resize after it is already
shared (`vision/preprocess.mojo`). No OpenCV at runtime: the Jetson's deploy
build runs this Mojo, and OpenCV is only the CALIBRATION
(`opencv.fisheye_calibrate`) and this file's test ORACLE
(`opencv.fisheye_project`, `opencv.fisheye_undistort_map`,
`tests/vision/test_fisheye.mojo`).

## The lens model (OpenCV's `cv::fisheye`, Kannala-Brandt, 4 terms)

For a camera-frame ray (x, y, z), z > 0, with a = x/z, b = y/z:

    r = sqrt(a^2 + b^2),  theta = atan(r)
    theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)
    u = fx * a * theta_d / r + cx,   v = fy * b * theta_d / r + cy

(skew 0 — the calibration fixes it). Pixel coordinates are OpenCV's: integer
coordinates are pixel CENTRES.

## The target pinhole (`Pinhole.sim`) — the tracer's convention, exactly

`raytrace/camera.camera_sample_ray` puts pixel (px, py)'s centre at
`u = (px + 0.5) / W` on a plane spanning `±tan(fovy/2) * W/H` horizontally
and `±tan(fovy/2)` vertically, row 0 at the TOP. In OpenCV pixel
coordinates that is

    fx = fy = H / (2 tan(fovy / 2)),   cx = W/2 - 0.5,   cy = H/2 - 0.5

The gate checks this against the tracer's own function, not against a
re-derivation.

## ⚠ Out-of-lens pixels are BLACK, and counted

An output pixel whose source falls outside the fisheye frame has no data.
`UndistortMap.n_outside` counts them; at the rig's 73.74 degrees inside a
170-degree lens it should be 0, and the importer refuses a map where it is
not, because a black band the sim never renders is a new domain gap.
"""

from std.math import atan, sqrt, tan, pi, floor

from .calib_file import CameraCalib


@fieldwise_init
struct FisheyeLens(Copyable, ImplicitlyCopyable, Movable, Writable):
    """A calibrated fisheye camera at its calibration resolution."""

    var fx: Float64
    var fy: Float64
    var cx: Float64
    var cy: Float64
    var k1: Float64
    var k2: Float64
    var k3: Float64
    var k4: Float64
    var width: Int
    var height: Int

    @staticmethod
    def from_calib(c: CameraCalib) raises -> Self:
        if c.model != "fisheye":
            raise Error(
                "FisheyeLens: calibration '" + c.name + "' is model '"
                + c.model + "', not fisheye — a radial-tangential dist vector"
                " read as Kannala-Brandt terms is a different lens"
            )
        if len(c.dist) != 4:
            raise Error(
                "FisheyeLens: a fisheye calibration has 4 terms, '" + c.name
                + "' has " + String(len(c.dist))
            )
        return Self(
            c.fx, c.fy, c.cx, c.cy, c.dist[0], c.dist[1], c.dist[2],
            c.dist[3], c.width, c.height,
        )

    def project(self, a: Float64, b: Float64) -> Tuple[Float64, Float64]:
        """The pixel of the ray `(a, b, 1)` — see the module header."""
        var r = sqrt(a * a + b * b)
        var scale = 1.0
        if r > 1e-12:
            var th = atan(r)
            var t2 = th * th
            var thd = th * (
                1.0 + t2 * (self.k1 + t2 * (self.k2 + t2 * (self.k3 + t2 * self.k4)))
            )
            scale = thd / r
        return (self.fx * a * scale + self.cx, self.fy * b * scale + self.cy)

    def unproject(self, u: Float64, v: Float64) raises -> Tuple[Float64, Float64]:
        """The ray `(a, b, 1)` pixel `(u, v)` sees — `project`'s inverse.

        Solves `theta_d = theta (1 + k1 theta^2 + ...)` for `theta` by Newton
        from `theta = theta_d` (the lens is near-equidistant), then
        `(a, b) = tan(theta) * (x', y') / theta_d`. What the extrinsics fit
        feeds a pinhole `solve_pnp` with `K = I`: the marker corners as
        undistorted normalised coordinates. Raises past 89 degrees (no
        pinhole ray) or if Newton does not converge."""
        var xp = (u - self.cx) / self.fx
        var yp = (v - self.cy) / self.fy
        var thd = sqrt(xp * xp + yp * yp)
        if thd < 1e-12:
            return (xp, yp)
        var th = thd
        for _ in range(30):
            var t2 = th * th
            var f = th * (
                1.0 + t2 * (self.k1 + t2 * (self.k2 + t2 * (self.k3 + t2 * self.k4)))
            ) - thd
            var df = 1.0 + t2 * (
                3.0 * self.k1 + t2 * (5.0 * self.k2 + t2 * (7.0 * self.k3 + t2 * 9.0 * self.k4))
            )
            var step = f / df
            th -= step
            if abs(step) < 1e-14:
                break
        if not (th > 0.0 and th < 89.0 * pi / 180.0):
            raise Error(
                "FisheyeLens.unproject: pixel (" + String(u) + ", " + String(v)
                + ") is not a ray in front of a pinhole (theta " + String(th) + ")"
            )
        var r = tan(th)
        return (xp * r / thd, yp * r / thd)

    def k_matrix(self) -> List[Float64]:
        return [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]

    def d_vector(self) -> List[Float64]:
        return [self.k1, self.k2, self.k3, self.k4]

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "fisheye ", self.width, "x", self.height, " f=(", self.fx, ", ",
            self.fy, ") c=(", self.cx, ", ", self.cy, ") k=(", self.k1, ", ",
            self.k2, ", ", self.k3, ", ", self.k4, ")",
        )


@fieldwise_init
struct Pinhole(Copyable, ImplicitlyCopyable, Movable):
    """An ideal pinhole in OpenCV pixel coordinates."""

    var fx: Float64
    var fy: Float64
    var cx: Float64
    var cy: Float64
    var width: Int
    var height: Int

    @staticmethod
    def sim(fovy_deg: Float64, width: Int, height: Int) -> Self:
        """The tracer's camera — `camera_sample_ray`'s convention, see the
        module header."""
        var f = Float64(height) / (2.0 * tan(0.5 * fovy_deg * pi / 180.0))
        return Self(
            f, f, Float64(width) / 2.0 - 0.5, Float64(height) / 2.0 - 0.5,
            width, height,
        )

    def k_matrix(self) -> List[Float64]:
        return [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]


struct UndistortMap(Movable):
    """For each OUTPUT (pinhole) pixel, the SOURCE (fisheye) pixel — built
    once per camera, applied per frame."""

    var out_w: Int
    var out_h: Int
    var src_w: Int
    var src_h: Int
    var map_x: List[Float32]
    var map_y: List[Float32]
    var n_outside: Int
    """Output pixels whose source is not inside the frame (rendered black)."""

    def __init__(out self, lens: FisheyeLens, pin: Pinhole):
        self.out_w = pin.width
        self.out_h = pin.height
        self.src_w = lens.width
        self.src_h = lens.height
        self.map_x = List[Float32](length=pin.width * pin.height, fill=0.0)
        self.map_y = List[Float32](length=pin.width * pin.height, fill=0.0)
        self.n_outside = 0
        for y in range(pin.height):
            var b = (Float64(y) - pin.cy) / pin.fy
            for x in range(pin.width):
                var a = (Float64(x) - pin.cx) / pin.fx
                var uv = lens.project(a, b)
                var i = y * pin.width + x
                self.map_x[i] = Float32(uv[0])
                self.map_y[i] = Float32(uv[1])
                if not self._inside(uv[0], uv[1]):
                    self.n_outside += 1

    @always_inline
    def _inside(self, sx: Float64, sy: Float64) -> Bool:
        return (
            sx >= 0.0 and sy >= 0.0 and sx <= Float64(self.src_w - 1)
            and sy <= Float64(self.src_h - 1)
        )

    def apply_hwc(
        self,
        ref src: List[UInt8],
        src_off: Int,
        channels: Int,
        mut dst: List[UInt8],
        dst_off: Int,
    ) raises:
        """Bilinear remap of one `[src_h, src_w, channels]` uint8 frame into
        `[out_h, out_w, channels]` at `dst_off`. Out-of-frame pixels are 0."""
        if len(src) < src_off + self.src_w * self.src_h * channels:
            raise Error("UndistortMap.apply_hwc: source frame is short")
        if len(dst) < dst_off + self.out_w * self.out_h * channels:
            raise Error("UndistortMap.apply_hwc: destination is short")
        var sw = self.src_w
        for i in range(self.out_w * self.out_h):
            var sx = Float64(self.map_x[i])
            var sy = Float64(self.map_y[i])
            var o = dst_off + i * channels
            if not self._inside(sx, sy):
                for c in range(channels):
                    dst[o + c] = 0
                continue
            var x0 = Int(floor(sx))
            var y0 = Int(floor(sy))
            var x1 = x0 + 1 if x0 + 1 < sw else x0
            var y1 = y0 + 1 if y0 + 1 < self.src_h else y0
            var fx = sx - Float64(x0)
            var fy = sy - Float64(y0)
            var w00 = (1.0 - fx) * (1.0 - fy)
            var w10 = fx * (1.0 - fy)
            var w01 = (1.0 - fx) * fy
            var w11 = fx * fy
            var p00 = src_off + (y0 * sw + x0) * channels
            var p10 = src_off + (y0 * sw + x1) * channels
            var p01 = src_off + (y1 * sw + x0) * channels
            var p11 = src_off + (y1 * sw + x1) * channels
            for c in range(channels):
                var v = (
                    w00 * Float64(Int(src[p00 + c]))
                    + w10 * Float64(Int(src[p10 + c]))
                    + w01 * Float64(Int(src[p01 + c]))
                    + w11 * Float64(Int(src[p11 + c]))
                )
                var q = Int(v + 0.5)
                dst[o + c] = UInt8(255 if q > 255 else (0 if q < 0 else q))
