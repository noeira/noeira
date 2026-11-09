"""A depth camera as DATA: pinhole intrinsics, a range window, an error model.

Same idea as `lidar.mojo`: a camera model is a `DepthCameraSpec`, normally a
text file in `zoo/` (`*.depth`), and it mounts on any body through the same
`LidarMount` (a pose in a body's frame). The scan reuses the lidar's ray cast;
what is new is the pixel pattern and the error model.

THE IMAGE: `width x height`, row 0 at the TOP, column 0 at the LEFT, OpenCV
intrinsics (`fx fy cx cy` in pixels, pixel CENTRES at integer coordinates — a
centred principal point is `cx = (width - 1) / 2`). That is what a real
calibration file holds (`vision/`), so a measured camera drops in as is.

THE SENSOR FRAME is the lidar's (REP-103: +x forward, +y left, +z up). Pixel
`(u, v)` looks along `(1, -(u - cx)/fx, -(v - cy)/fy)` — right in the image is
-y, down is -z. ⚠ NOT MuJoCo's camera frame (it looks down its own -Z with +Y
up); `depth_mount_from_mjcf_camera` converts one to the other.

THE OUTPUT is PLANAR depth (distance along the optical axis, what a D435 or a
ToF camera reports), **0 = no depth**, as the real drivers and the camera
renderer report it. ⚠ Not the ray length: a pixel at the image corner sees a
wall at the same depth as the centre pixel does, through a longer ray.

THE ERROR MODELS (`model =`):

  ideal    exact depth (the gate runs this)
  stereo   active/passive stereo: sigma_z = z^2 * subpixel / (fx * baseline)
           — a disparity error of `subpixel` pixels, turned into depth. Grows
           with the SQUARE of the distance: Intel quotes < 2 % at 2 m for a D435.
  tof      time of flight: sigma_z = sigma_a + sigma_b * z

plus, for every model but `ideal`: dropout, a grazing-angle cut (a surface
seen beyond `max_incidence_deg` from its normal returns nothing), depth
quantised to `depth_unit`, and for `stereo` with `stereo_shadow = 1` the
OCCLUSION SHADOW — a pixel the second imager (at `baseline` to the right)
cannot see has no depth. That is the band of missing depth a D435 shows on the
LEFT of every foreground object, `fx * baseline * (1/z_near - 1/z_far)` pixels
wide.

⚠ NOT MODELLED: dark / specular / transparent surfaces (no material on a ray),
the projector pattern, multipath for ToF, motion blur, rolling shutter.
"""

from std.math import cos, sin, sqrt, log, pi, tan, atan, floor

from .spec_text import spec_lines, read_text
from .lidar import lidar_uniforms

comptime DEPTH_IDEAL: Int = 0
comptime DEPTH_STEREO: Int = 1
comptime DEPTH_TOF: Int = 2


struct DepthCameraSpec(Copyable, Movable, Writable):
    var name: String
    var width: Int
    var height: Int
    var fx: Float64
    var fy: Float64
    var cx: Float64
    var cy: Float64
    var range_min: Float64
    var range_max: Float64
    var model: Int
    var baseline: Float64
    """Stereo baseline, metres (stereo model and the occlusion shadow)."""
    var subpixel: Float64
    """Disparity error, pixels (stereo model)."""
    var sigma_a: Float64
    var sigma_b: Float64
    """ToF: sigma_z = sigma_a + sigma_b * z."""
    var depth_unit: Float64
    """Quantisation step, metres; 0 = none."""
    var max_incidence_deg: Float64
    """Beyond this angle from the surface normal: no depth. 90 = never cut."""
    var dropout: Float64
    var stereo_shadow: Bool
    var rate_hz: Float64

    def __init__(out self):
        self.name = String("unnamed")
        self.width = 0
        self.height = 0
        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0
        self.range_min = 0.0
        self.range_max = 1.0e6
        self.model = DEPTH_IDEAL
        self.baseline = 0.0
        self.subpixel = 0.0
        self.sigma_a = 0.0
        self.sigma_b = 0.0
        self.depth_unit = 0.0
        self.max_incidence_deg = 90.0
        self.dropout = 0.0
        self.stereo_shadow = False
        self.rate_hz = 30.0

    def npix(self) -> Int:
        return self.width * self.height

    def ideal(self) -> Self:
        """The same camera with every error switched off and the window open."""
        var s = self.copy()
        s.model = DEPTH_IDEAL
        s.range_min = 0.0
        s.range_max = 1.0e30
        s.depth_unit = 0.0
        s.max_incidence_deg = 90.0
        s.dropout = 0.0
        s.stereo_shadow = False
        return s^

    def validate(self) raises:
        var who = String("depth spec '") + self.name + "': "
        if self.width < 1 or self.height < 1:
            raise Error(who + "width and height must be >= 1")
        if not (self.fx > 0.0 and self.fy > 0.0):
            raise Error(who + "fx and fy must be > 0")
        if not (self.range_max > self.range_min and self.range_min >= 0.0):
            raise Error(who + "need 0 <= range_min < range_max")
        if self.model < DEPTH_IDEAL or self.model > DEPTH_TOF:
            raise Error(who + "unknown model")
        if (self.model == DEPTH_STEREO or self.stereo_shadow) and not (self.baseline > 0.0):
            raise Error(who + "stereo needs baseline > 0")
        if self.stereo_shadow and self.model != DEPTH_STEREO:
            raise Error(who + "stereo_shadow needs model = stereo")
        if self.subpixel < 0.0 or self.sigma_a < 0.0 or self.sigma_b < 0.0:
            raise Error(who + "noise terms must be >= 0")
        if self.depth_unit < 0.0:
            raise Error(who + "depth_unit must be >= 0")
        if not (self.max_incidence_deg > 0.0 and self.max_incidence_deg <= 90.0):
            raise Error(who + "max_incidence_deg must be in (0, 90]")
        if not (self.dropout >= 0.0 and self.dropout < 1.0):
            raise Error(who + "dropout must be in [0, 1)")

    def hfov_deg(self) -> Float64:
        """Horizontal field of view between the outer pixel EDGES."""
        var l = (-0.5 - self.cx) / self.fx
        var r = (Float64(self.width) - 0.5 - self.cx) / self.fx
        return (atan(r) - atan(l)) * 180.0 / pi

    def write_to(self, mut writer: Some[Writer]):
        var m = String("ideal")
        if self.model == DEPTH_STEREO:
            m = "stereo"
        elif self.model == DEPTH_TOF:
            m = "tof"
        writer.write(
            "DepthCameraSpec(", self.name, ", ", self.width, "x", self.height,
            ", fx ", self.fx, ", ", m, ", ", self.range_min, "-",
            self.range_max, " m, shadow ", self.stereo_shadow, ", ",
            self.rate_hz, " Hz)",
        )


def _model_of(v: String) raises -> Int:
    if v == "ideal":
        return DEPTH_IDEAL
    if v == "stereo":
        return DEPTH_STEREO
    if v == "tof":
        return DEPTH_TOF
    raise Error("depth spec: model must be ideal, stereo or tof, not '" + v + "'")


def depth_spec_from_text(text: String) raises -> DepthCameraSpec:
    """`key = value` lines (see `spec_text`). An unknown key raises."""
    var s = DepthCameraSpec()
    for ln in spec_lines(text, "depth"):
        var k = ln.key
        var v = ln.value
        if k == "name":
            s.name = v
        elif k == "width":
            s.width = atol(v)
        elif k == "height":
            s.height = atol(v)
        elif k == "fx":
            s.fx = atof(v)
        elif k == "fy":
            s.fy = atof(v)
        elif k == "cx":
            s.cx = atof(v)
        elif k == "cy":
            s.cy = atof(v)
        elif k == "range_min":
            s.range_min = atof(v)
        elif k == "range_max":
            s.range_max = atof(v)
        elif k == "model":
            s.model = _model_of(v)
        elif k == "baseline":
            s.baseline = atof(v)
        elif k == "subpixel":
            s.subpixel = atof(v)
        elif k == "sigma_a":
            s.sigma_a = atof(v)
        elif k == "sigma_b":
            s.sigma_b = atof(v)
        elif k == "depth_unit":
            s.depth_unit = atof(v)
        elif k == "max_incidence_deg":
            s.max_incidence_deg = atof(v)
        elif k == "dropout":
            s.dropout = atof(v)
        elif k == "stereo_shadow":
            s.stereo_shadow = atol(v) != 0
        elif k == "rate_hz":
            s.rate_hz = atof(v)
        else:
            raise Error("depth spec: unknown key '" + k + "'")
    s.validate()
    return s^


def load_depth_spec(path: String) raises -> DepthCameraSpec:
    return depth_spec_from_text(read_text(path))


def depth_spec_pinhole_fovy(
    width: Int, height: Int, fovy_deg: Float64, name: String = "pinhole"
) raises -> DepthCameraSpec:
    """An ideal camera with MuJoCo's `<camera fovy>` geometry: square pixels,
    the principal point at the image centre, `fy = (height/2) / tan(fovy/2)`.
    """
    var s = DepthCameraSpec()
    s.name = name
    s.width = width
    s.height = height
    s.fy = 0.5 * Float64(height) / tan(0.5 * fovy_deg * pi / 180.0)
    s.fx = s.fy
    s.cx = 0.5 * Float64(width - 1)
    s.cy = 0.5 * Float64(height - 1)
    s.validate()
    return s^


def depth_directions(spec: DepthCameraSpec) -> List[Float64]:
    """Per pixel, row-major from the top-left: the UNIT ray in the sensor
    frame (3 floats). Its x component is the factor from ray length to planar
    depth: `z = t * d.x`."""
    var out = List[Float64]()
    out.reserve(3 * spec.npix())
    for v in range(spec.height):
        var yo = (Float64(v) - spec.cy) / spec.fy
        for u in range(spec.width):
            var xo = (Float64(u) - spec.cx) / spec.fx
            var inv = 1.0 / sqrt(1.0 + xo * xo + yo * yo)
            out.append(inv)
            out.append(-xo * inv)
            out.append(-yo * inv)
    return out^


def depth_to_points(spec: DepthCameraSpec, depth: List[Float64]) -> List[Float64]:
    """Unproject one image to points in the SENSOR frame, 3 floats per valid
    pixel (0-depth pixels are skipped). `p = z * (1, -(u-cx)/fx, -(v-cy)/fy)`."""
    var out = List[Float64]()
    for v in range(spec.height):
        var yo = (Float64(v) - spec.cy) / spec.fy
        for u in range(spec.width):
            var z = depth[v * spec.width + u]
            if z <= 0.0:
                continue
            out.append(z)
            out.append(-z * (Float64(u) - spec.cx) / spec.fx)
            out.append(-z * yo)
    return out^


@always_inline
def depth_return[
    DTYPE: DType
](
    z: Scalar[DTYPE],
    cos_inc: Scalar[DTYPE],
    shadowed: Bool,
    model: Int,
    range_min: Scalar[DTYPE],
    range_max: Scalar[DTYPE],
    k_stereo: Scalar[DTYPE],
    sigma_a: Scalar[DTYPE],
    sigma_b: Scalar[DTYPE],
    cos_max_inc: Scalar[DTYPE],
    unit: Scalar[DTYPE],
    dropout: Scalar[DTYPE],
    seed: UInt64,
    frame: Int,
    lane: Int,
    pix: Int,
) -> Scalar[DTYPE] where DTYPE.is_floating_point():
    """True planar depth `z` (<= 0 = no hit) to what the camera reports (0 =
    no depth). `k_stereo = subpixel / (fx * baseline)`; `cos_inc` = |cos| of
    the angle between the ray and the surface normal.

    Order: geometry cuts (shadow, grazing), dropout, noise, window, then
    quantisation — a noisy depth outside the window is lost, as on a device.
    """
    var none = Scalar[DTYPE](0)
    if z <= 0:
        return none
    if model == DEPTH_IDEAL:
        if z < range_min or z > range_max:
            return none
        return z
    if shadowed or cos_inc < cos_max_inc:
        return none
    var u = lidar_uniforms[DTYPE](seed, frame, lane, pix)
    if u[0] < dropout:
        return none
    var sigma = sigma_a + sigma_b * z
    if model == DEPTH_STEREO:
        sigma = k_stereo * z * z
    var r = z
    if sigma > 0:
        r = z + sigma * sqrt(Scalar[DTYPE](-2.0) * log(u[1])) * cos(
            Scalar[DTYPE](2.0 * pi) * u[2]
        )
    if r < range_min or r > range_max:
        return none
    if unit > 0:
        r = floor(r / unit + Scalar[DTYPE](0.5)) * unit
    return r
