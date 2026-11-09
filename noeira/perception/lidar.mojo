"""A lidar as DATA: a spec (scan pattern, range limits, noise) and a mount.

No sensor is hard-coded here. A lidar model is a `LidarSpec`, normally read
from a small text file (`zoo/*.lidar`), so trying a different sensor is a new
file and not new code — the sim is meant to choose the hardware, not to be
written around one.

THE SENSOR FRAME (REP-103): +x forward, +y left, +z up. A ray at azimuth `az`
(counter-clockwise from +x) and elevation `el` (up from the xy plane) points
along `(cos el cos az, cos el sin az, sin el)`. ⚠ This is NOT the MuJoCo
rangefinder's convention (the site's +Z) and not a camera's (-Z): a lidar's
mount frame is its own body frame, x forward and z up, as the drivers report it.

THREE PATTERNS cover the zoo:

  fan     one plane (el = 0), `h_samples` azimuths — an RPLidar-class 2D lidar
  rings   `v_samples` rings evenly spaced in [v_min, v_max], `h_samples`
          azimuths each — a spinning multi-beam lidar (VLP-16-class)
  spiral  `points` directions per frame from a low-discrepancy sequence over
          the band, uniform in SOLID ANGLE, continuing frame after frame so
          coverage fills in over time — a non-repetitive scanner (MID-360-class)

⚠ `spiral` IS AN APPROXIMATION, not Livox's pattern. The real rosette is
denser in some parts of the band; this one is uniform. It reproduces the two
properties a mapper sees — a fixed point budget per frame and coverage that
keeps filling in — not the exact density.

A RETURN: the distance along the (unit) ray in metres, or `LIDAR_NO_RETURN`
(-1) when nothing is hit inside [range_min, range_max] or the beam dropped
out. ⚠ -1 is a sentinel, as for MuJoCo's rangefinder: a mapper must read it
as "free up to range_max", never as a distance.
"""

from std.math import cos, sin, sqrt, log, pi, asin

from .spec_text import spec_lines, read_text

comptime LIDAR_FAN: Int = 0
comptime LIDAR_RINGS: Int = 1
comptime LIDAR_SPIRAL: Int = 2

comptime LIDAR_NO_RETURN: Float64 = -1.0

# The R2 low-discrepancy sequence (Roberts 2018): the plastic number `g`,
# `a1 = 1/g`, `a2 = 1/g^2`. Successive points never repeat and fill the unit
# square evenly, which is the property `spiral` needs.
comptime _R2_A1: Float64 = 0.7548776662466927
comptime _R2_A2: Float64 = 0.5698402909980532


def _frac(x: Float64) -> Float64:
    return x - Float64(Int(x)) if x >= 0 else x - Float64(Int(x)) + 1.0


struct LidarSpec(Copyable, Movable, Writable):
    """One lidar model. Angles in degrees, distances in metres."""

    var name: String
    var pattern: Int
    var h_fov_deg: Float64
    """Horizontal field of view, centred on +x. 360 = all round."""
    var h_samples: Int
    """Azimuths per ring (fan, rings)."""
    var v_min_deg: Float64
    var v_max_deg: Float64
    var v_samples: Int
    """Rings (rings only)."""
    var points: Int
    """Directions per frame (spiral only)."""
    var range_min: Float64
    var range_max: Float64
    var range_sigma: Float64
    """Gaussian range noise, metres (constant; a datasheet's % of range is
    not modelled yet)."""
    var dropout: Float64
    """Probability that a beam returns nothing, in [0, 1)."""
    var rate_hz: Float64
    """Frames per second — metadata for a consumer's scheduling."""

    def __init__(out self):
        self.name = String("unnamed")
        self.pattern = LIDAR_FAN
        self.h_fov_deg = 360.0
        self.h_samples = 360
        self.v_min_deg = 0.0
        self.v_max_deg = 0.0
        self.v_samples = 1
        self.points = 0
        self.range_min = 0.0
        self.range_max = 1.0e6
        self.range_sigma = 0.0
        self.dropout = 0.0
        self.rate_hz = 10.0

    def rays_per_frame(self) -> Int:
        if self.pattern == LIDAR_FAN:
            return self.h_samples
        if self.pattern == LIDAR_RINGS:
            return self.h_samples * self.v_samples
        return self.points

    def validate(self) raises:
        """Refuse a spec that would scan something other than it says."""
        var who = String("lidar spec '") + self.name + "': "
        if self.pattern < LIDAR_FAN or self.pattern > LIDAR_SPIRAL:
            raise Error(who + "unknown pattern " + String(self.pattern))
        if not (self.h_fov_deg > 0.0 and self.h_fov_deg <= 360.0):
            raise Error(who + "h_fov_deg must be in (0, 360]")
        if self.pattern != LIDAR_SPIRAL and self.h_samples < 1:
            raise Error(who + "h_samples must be >= 1")
        if self.pattern == LIDAR_RINGS and self.v_samples < 2:
            raise Error(who + "rings needs v_samples >= 2 (one ring is a fan)")
        if self.pattern != LIDAR_FAN and not (self.v_max_deg > self.v_min_deg):
            raise Error(who + "v_max_deg must be > v_min_deg")
        if self.v_min_deg < -90.0 or self.v_max_deg > 90.0:
            raise Error(who + "elevations must lie in [-90, 90]")
        if self.pattern == LIDAR_SPIRAL and self.points < 1:
            raise Error(who + "spiral needs points >= 1")
        if not (self.range_max > self.range_min and self.range_min >= 0.0):
            raise Error(who + "need 0 <= range_min < range_max")
        if self.range_sigma < 0.0:
            raise Error(who + "range_sigma must be >= 0")
        if not (self.dropout >= 0.0 and self.dropout < 1.0):
            raise Error(who + "dropout must be in [0, 1)")

    def write_to(self, mut writer: Some[Writer]):
        var p = String("fan")
        if self.pattern == LIDAR_RINGS:
            p = "rings"
        elif self.pattern == LIDAR_SPIRAL:
            p = "spiral"
        writer.write(
            "LidarSpec(", self.name, ", ", p, ", ", self.rays_per_frame(),
            " rays/frame, ", self.range_min, "-", self.range_max, " m, sigma ",
            self.range_sigma, ", dropout ", self.dropout, ", ", self.rate_hz,
            " Hz)",
        )


def _pattern_of(v: String) raises -> Int:
    if v == "fan":
        return LIDAR_FAN
    if v == "rings":
        return LIDAR_RINGS
    if v == "spiral":
        return LIDAR_SPIRAL
    raise Error("lidar spec: pattern must be fan, rings or spiral, not '" + v + "'")


def lidar_spec_from_text(text: String) raises -> LidarSpec:
    """Parse `key = value` lines; `#` starts a comment.

    ⚠ AN UNKNOWN KEY RAISES. A misspelt `range_sigam = 0.02` that was silently
    ignored would run a noiseless experiment labelled as a noisy one.
    """
    var s = LidarSpec()
    var seen_pattern = False
    for ln in spec_lines(text, "lidar"):
        var k = ln.key
        var v = ln.value
        if k == "name":
            s.name = v
        elif k == "pattern":
            s.pattern = _pattern_of(v)
            seen_pattern = True
        elif k == "h_fov_deg":
            s.h_fov_deg = atof(v)
        elif k == "h_samples":
            s.h_samples = atol(v)
        elif k == "v_min_deg":
            s.v_min_deg = atof(v)
        elif k == "v_max_deg":
            s.v_max_deg = atof(v)
        elif k == "v_samples":
            s.v_samples = atol(v)
        elif k == "points":
            s.points = atol(v)
        elif k == "range_min":
            s.range_min = atof(v)
        elif k == "range_max":
            s.range_max = atof(v)
        elif k == "range_sigma":
            s.range_sigma = atof(v)
        elif k == "dropout":
            s.dropout = atof(v)
        elif k == "rate_hz":
            s.rate_hz = atof(v)
        else:
            raise Error("lidar spec: unknown key '" + k + "'")
    if not seen_pattern:
        raise Error("lidar spec '" + s.name + "': no `pattern =` line")
    s.validate()
    return s^


def load_lidar_spec(path: String) raises -> LidarSpec:
    return lidar_spec_from_text(read_text(path))


def _azimuth(spec: LidarSpec, i: Int, n: Int) -> Float64:
    """Azimuth `i` of `n`, radians. All round: no duplicate endpoint. A partial
    FOV: both edges included, so the fan spans exactly the FOV."""
    var fov = spec.h_fov_deg * pi / 180.0
    if n == 1:
        return 0.0
    if spec.h_fov_deg >= 360.0:
        return -pi + fov * Float64(i) / Float64(n)
    return -0.5 * fov + fov * Float64(i) / Float64(n - 1)


def lidar_directions(spec: LidarSpec, frame: Int = 0) -> List[Float64]:
    """Unit ray directions in the SENSOR frame, `3 * rays_per_frame()` floats.

    Ring-major for `rings` (all azimuths of the lowest ring first). `frame`
    only matters for `spiral`, whose sequence continues from frame to frame.
    """
    var out = List[Float64]()
    var n = spec.rays_per_frame()
    out.reserve(3 * n)
    if spec.pattern == LIDAR_FAN:
        for i in range(n):
            var az = _azimuth(spec, i, n)
            out.append(cos(az))
            out.append(sin(az))
            out.append(0.0)
    elif spec.pattern == LIDAR_RINGS:
        var v0 = spec.v_min_deg * pi / 180.0
        var v1 = spec.v_max_deg * pi / 180.0
        for r in range(spec.v_samples):
            var el = v0 + (v1 - v0) * Float64(r) / Float64(spec.v_samples - 1)
            var ce = cos(el)
            var se = sin(el)
            for i in range(spec.h_samples):
                var az = _azimuth(spec, i, spec.h_samples)
                out.append(ce * cos(az))
                out.append(ce * sin(az))
                out.append(se)
    else:
        # Uniform in solid angle over the band: sin(el) uniform in
        # [sin v_min, sin v_max], azimuth uniform over the FOV.
        var s0 = sin(spec.v_min_deg * pi / 180.0)
        var s1 = sin(spec.v_max_deg * pi / 180.0)
        var fov = spec.h_fov_deg * pi / 180.0
        for k in range(n):
            var idx = Float64(frame * n + k + 1)
            var u = _frac(0.5 + _R2_A1 * idx)
            var w = _frac(0.5 + _R2_A2 * idx)
            var az = -0.5 * fov + fov * u
            var se = s0 + (s1 - s0) * w
            var ce = sqrt(max(0.0, 1.0 - se * se))
            out.append(ce * cos(az))
            out.append(ce * sin(az))
            out.append(se)
    return out^


@fieldwise_init
struct LidarMount(Copyable, ImplicitlyCopyable, Movable, Writable):
    """Where the sensor sits: a pose in the frame of ANY body of ANY model.

    `body` is the model's body index; `pos` and the unit quaternion
    (`qw, qx, qy, qz`) are the sensor frame in that body's frame. Nothing
    reads a robot's names, so the same spec mounts on a G1 torso, a BHL head
    or a ToddlerBot at whatever height is being priced.

    ⚠ THE MOUNT BODY IS EXCLUDED FROM ITS OWN SCAN, AND NOTHING ELSE IS —
    MuJoCo's `bodyexclude` rule, as for the rangefinder. A head lidar sees
    the robot's own arms, as a real one does; filtering self-hits is the
    mapper's job, not the sensor's.
    """

    var body: Int
    var px: Float64
    var py: Float64
    var pz: Float64
    var qw: Float64
    var qx: Float64
    var qy: Float64
    var qz: Float64

    @staticmethod
    def on_body(body: Int, px: Float64, py: Float64, pz: Float64) -> Self:
        """Level with the body frame, offset by `(px, py, pz)`."""
        return Self(body, px, py, pz, 1.0, 0.0, 0.0, 0.0)


# ── noise: one counter-based draw per (seed, frame, lane, ray) ───────────────
#
# ⚠ COUNTER-BASED, NOT A STREAM. A stream's draws depend on the ORDER rays are
# visited, so a GPU launch (any order) and a CPU loop would disagree; a hash of
# the ray's identity gives every target the same noise for the same ray.


@always_inline
def _mix64(x0: UInt64) -> UInt64:
    """splitmix64's finaliser."""
    var x = x0
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB
    return x ^ (x >> 31)


@always_inline
def lidar_uniforms[
    DTYPE: DType
](seed: UInt64, frame: Int, lane: Int, ray: Int) -> Tuple[
    Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]
] where DTYPE.is_floating_point():
    """Three independent uniforms in (0, 1) for one ray of one frame.

    ⚠ 24 BITS EACH, so the value is EXACT in float32 and float64 alike: a CPU
    float64 scan and a GPU float32 scan draw the same uniforms. (Metal has no
    double, so the draw cannot go through Float64.)
    """
    var h = _mix64(seed ^ 0x9E3779B97F4A7C15)
    h = _mix64(h ^ UInt64(frame))
    h = _mix64(h ^ UInt64(lane))
    h = _mix64(h ^ UInt64(ray))
    var a = _mix64(h ^ 1)
    var b = _mix64(h ^ 2)
    var c = _mix64(h ^ 3)
    comptime S = 1.0 / 16777216.0  # 2^-24
    return (
        (Scalar[DTYPE](Int(a >> 40)) + 0.5) * S,
        (Scalar[DTYPE](Int(b >> 40)) + 0.5) * S,
        (Scalar[DTYPE](Int(c >> 40)) + 0.5) * S,
    )


@always_inline
def lidar_return[
    DTYPE: DType
](
    t: Scalar[DTYPE],
    range_min: Scalar[DTYPE],
    range_max: Scalar[DTYPE],
    sigma: Scalar[DTYPE],
    dropout: Scalar[DTYPE],
    seed: UInt64,
    frame: Int,
    lane: Int,
    ray: Int,
) -> Scalar[DTYPE] where DTYPE.is_floating_point():
    """A raw ray distance `t` (-1 = miss) to what the sensor reports.

    Order: dropout, then range noise, then the range window — a noisy return
    that lands outside the window is lost, as on a real unit.
    """
    var none = Scalar[DTYPE](LIDAR_NO_RETURN)
    if t < 0:
        return none
    var r = t
    if dropout > 0 or sigma > 0:
        var u = lidar_uniforms[DTYPE](seed, frame, lane, ray)
        if u[0] < dropout:
            return none
        if sigma > 0:
            # Box-Muller, one of the pair.
            r = t + sigma * sqrt(Scalar[DTYPE](-2.0) * log(u[1])) * cos(
                Scalar[DTYPE](2.0 * pi) * u[2]
            )
    if r < range_min or r > range_max:
        return none
    return r
