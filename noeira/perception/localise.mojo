"""2D localisation: drifting odometry, corrected by matching scans to a map.

    var odo = Odometry(spec, seed)                       # per episode
    odo.step(vx_true, vy_true, wz_true, dt)              # body-frame twist in
    var lf = LikelihoodField(occ, x0, y0, res, nx, ny, sigma=0.05)
    var m = correlative_match(lf, pts_xy, prior_x, prior_y, prior_yaw)

WHAT IS ESTIMATED: the base's (x, y, yaw) on the floor plane. Roll, pitch,
height and the joint angles are taken as KNOWN — on a real robot the IMU's
gravity vector gives roll and pitch, leg kinematics the height, encoders the
joints — so a scan arrives as points in a GRAVITY-ALIGNED BASE FRAME (origin
under the base, x = the base's heading, z = height above the floor). What
drifts, and what a map has to fix, is position and heading.

ODOMETRY (`OdometrySpec`, data like every sensor): the true body-frame twist
corrupted the way leg odometry + a gyro are — a per-episode SCALE error on
the forward and lateral speed (foot slip, leg length), white noise on both,
and a per-episode gyro BIAS plus white noise on the yaw rate — integrated from
the episode's start pose. ⚠ It is a model of an estimator's error, not a leg
kinematics estimator; its numbers are ASSUMED (see `zoo/*.odom`).

SCAN MATCHING: Olson's correlative matcher (ICRA 2009) on a likelihood field —
each map cell scores exp(-d^2 / 2 sigma^2), d the distance to the nearest
occupied cell (an exact EDT) — searched exhaustively around the prior, coarse
then fine. Exhaustive, so no local minimum inside the window; the window is
what bounds how much drift one scan can fix.
"""

from std.math import cos, sin, sqrt, exp, log, pi, atan2

from noeira.planners.nav2d import cell_distance_transform
from .spec_text import spec_lines, read_text
from .lidar import lidar_uniforms


def wrap_angle(a: Float64) -> Float64:
    var x = a
    while x > pi:
        x -= 2.0 * pi
    while x < -pi:
        x += 2.0 * pi
    return x


struct OdometrySpec(Copyable, Movable, Writable):
    var name: String
    var scale_sigma: Float64
    """Per-episode relative scale error on forward and lateral speed (1 sigma)."""
    var lin_noise: Float64
    """White noise on each speed component, m/s (1 sigma, per step)."""
    var gyro_bias_sigma: Float64
    """Per-episode yaw-rate bias, rad/s (1 sigma)."""
    var gyro_noise: Float64
    """White noise on the yaw rate, rad/s (1 sigma, per step)."""

    def __init__(out self):
        self.name = String("ideal")
        self.scale_sigma = 0.0
        self.lin_noise = 0.0
        self.gyro_bias_sigma = 0.0
        self.gyro_noise = 0.0

    def write_to(self, mut writer: Some[Writer]):
        writer.write("OdometrySpec(", self.name, ", scale ", self.scale_sigma, ", v noise ",
                     self.lin_noise, " m/s, gyro bias ", self.gyro_bias_sigma, " rad/s, noise ",
                     self.gyro_noise, " rad/s)")


def odometry_spec_from_text(text: String) raises -> OdometrySpec:
    var s = OdometrySpec()
    for ln in spec_lines(text, "odometry"):
        var k = ln.key
        var v = ln.value
        if k == "name":
            s.name = v
        elif k == "scale_sigma":
            s.scale_sigma = atof(v)
        elif k == "lin_noise":
            s.lin_noise = atof(v)
        elif k == "gyro_bias_sigma":
            s.gyro_bias_sigma = atof(v)
        elif k == "gyro_noise":
            s.gyro_noise = atof(v)
        else:
            raise Error("odometry spec: unknown key '" + k + "'")
    if s.scale_sigma < 0 or s.lin_noise < 0 or s.gyro_bias_sigma < 0 or s.gyro_noise < 0:
        raise Error("odometry spec '" + s.name + "': sigmas must be >= 0")
    return s^


def load_odometry_spec(path: String) raises -> OdometrySpec:
    return odometry_spec_from_text(read_text(path))


def _gauss(seed: UInt64, a: Int, b: Int, c: Int) -> Float64:
    var u = lidar_uniforms[DType.float64](seed, a, b, c)
    return sqrt(-2.0 * log(u[0])) * cos(2.0 * pi * u[1])


struct Odometry(Movable):
    """The dead-reckoned pose: integrated, corrupted twist. `x, y, yaw` is the
    ESTIMATE; a localiser overwrites it after a match (`set_pose`)."""

    var spec: OdometrySpec
    var seed: UInt64
    var k: Int
    var scale_v: Float64
    var scale_l: Float64
    var bias_w: Float64
    var x: Float64
    var y: Float64
    var yaw: Float64
    var last_v: Float64
    var last_l: Float64
    var last_w: Float64
    """The last MEASURED twist — what a controller has for its own speed."""

    def __init__(out self, spec: OdometrySpec, seed: Int):
        self.spec = spec.copy()
        self.seed = UInt64(seed)
        self.k = 0
        self.scale_v = 1.0 + spec.scale_sigma * _gauss(self.seed, 1, 0, 0)
        self.scale_l = 1.0 + spec.scale_sigma * _gauss(self.seed, 2, 0, 0)
        self.bias_w = spec.gyro_bias_sigma * _gauss(self.seed, 3, 0, 0)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_v = 0.0
        self.last_l = 0.0
        self.last_w = 0.0

    def set_pose(mut self, x: Float64, y: Float64, yaw: Float64):
        self.x = x
        self.y = y
        self.yaw = wrap_angle(yaw)

    def measured_twist(mut self, vx: Float64, vy: Float64, wz: Float64) -> Tuple[Float64, Float64, Float64]:
        """The true body-frame twist as the odometry reports it, one step."""
        self.k += 1
        var mv = vx * self.scale_v + self.spec.lin_noise * _gauss(self.seed, 10, self.k, 0)
        var ml = vy * self.scale_l + self.spec.lin_noise * _gauss(self.seed, 11, self.k, 0)
        var mw = wz + self.bias_w + self.spec.gyro_noise * _gauss(self.seed, 12, self.k, 0)
        return (mv, ml, mw)

    def step(mut self, vx: Float64, vy: Float64, wz: Float64, dt: Float64):
        """Integrate one step of the MEASURED twist (midpoint heading)."""
        var m = self.measured_twist(vx, vy, wz)
        self.last_v = m[0]
        self.last_l = m[1]
        self.last_w = m[2]
        var mid = self.yaw + 0.5 * m[2] * dt
        self.x += (cos(mid) * m[0] - sin(mid) * m[1]) * dt
        self.y += (sin(mid) * m[0] + cos(mid) * m[1]) * dt
        self.yaw = wrap_angle(self.yaw + m[2] * dt)


struct LikelihoodField(Movable):
    """exp(-d^2 / 2 sigma^2) per cell, d = metres to the nearest occupied cell.
    Same box and cell layout as `VoxelMap.occupied_2d` / `NavGrid`."""

    var x0: Float64
    var y0: Float64
    var res: Float64
    var nx: Int
    var ny: Int
    var sigma: Float64
    var f: List[Float32]

    def __init__(
        out self, occ: List[Bool], x0: Float64, y0: Float64, res: Float64,
        nx: Int, ny: Int, sigma: Float64 = 0.05,
    ) raises:
        if len(occ) != nx * ny:
            raise Error("LikelihoodField: occ has " + String(len(occ)) + " cells, not nx*ny")
        if not (sigma > 0.0):
            raise Error("LikelihoodField: sigma must be > 0")
        self.x0 = x0
        self.y0 = y0
        self.res = res
        self.nx = nx
        self.ny = ny
        self.sigma = sigma
        var dc = cell_distance_transform(occ, nx, ny)
        self.f = List[Float32](length=nx * ny, fill=0)
        var k = 1.0 / (2.0 * sigma * sigma)
        for c in range(nx * ny):
            var d = dc[c] * res
            self.f[c] = Float32(exp(-d * d * k)) if d < 6.0 * sigma else Float32(0)

    @always_inline
    def at(self, x: Float64, y: Float64) -> Float32:
        var i = Int((x - self.x0) / self.res)
        var j = Int((y - self.y0) / self.res)
        if x < self.x0 or y < self.y0 or i >= self.nx or j >= self.ny:
            return 0
        return self.f[j * self.nx + i]


@fieldwise_init
struct Match(Copyable, ImplicitlyCopyable, Movable, Writable):
    var x: Float64
    var y: Float64
    var yaw: Float64
    var score: Float64
    """Mean likelihood of the points at the best pose, in [0, 1]."""
    var n: Int


def _search(
    lf: LikelihoodField, px: List[Float64], py: List[Float64],
    cx: Float64, cy: Float64, cyaw: Float64,
    wxy: Float64, sxy: Float64, wyaw: Float64, syaw: Float64,
) -> Match:
    var n = len(px)
    var best = Match(cx, cy, cyaw, -1.0, n)
    var nxy = Int(wxy / sxy + 0.5)
    var nyaw = Int(wyaw / syaw + 0.5)
    var rx = List[Float64](length=n, fill=0.0)
    var ry = List[Float64](length=n, fill=0.0)
    for a in range(-nyaw, nyaw + 1):
        var th = cyaw + Float64(a) * syaw
        var c = cos(th)
        var s = sin(th)
        for k in range(n):
            rx[k] = c * px[k] - s * py[k]
            ry[k] = s * px[k] + c * py[k]
        for bi in range(-nxy, nxy + 1):
            var tx = cx + Float64(bi) * sxy
            for bj in range(-nxy, nxy + 1):
                var ty = cy + Float64(bj) * sxy
                var acc: Float32 = 0
                for k in range(n):
                    acc += lf.at(rx[k] + tx, ry[k] + ty)
                var sc = Float64(acc) / Float64(max(1, n))
                if sc > best.score:
                    best = Match(tx, ty, wrap_angle(th), sc, n)
    return best


def correlative_match(
    lf: LikelihoodField,
    px: List[Float64],
    py: List[Float64],
    prior_x: Float64, prior_y: Float64, prior_yaw: Float64,
    window_xy: Float64 = 0.20,
    window_yaw: Float64 = 0.10,
) -> Match:
    """Best (x, y, yaw) of the base for points `(px, py)` in the gravity-aligned
    base frame, searched exhaustively within the window around the prior:
    4 cm / 1 deg steps, then 1 cm / 0.25 deg around the best."""
    var deg = pi / 180.0
    var c = _search(lf, px, py, prior_x, prior_y, prior_yaw, window_xy, 0.04, window_yaw, 1.0 * deg)
    return _search(lf, px, py, c.x, c.y, c.yaw, 0.04, 0.01, 1.0 * deg, 0.25 * deg)


def base_frame_points(
    endpoints: List[Float64],
    hits: List[Bool],
    base_x: Float64, base_y: Float64, base_yaw: Float64,
    z_lo: Float64, z_hi: Float64,
    max_points: Int,
    mut px: List[Float64],
    mut py: List[Float64],
):
    """A scan's HITS (world frame) as points in the gravity-aligned base frame,
    kept when their height lies in [z_lo, z_hi] (the band the map holds), at
    most `max_points` (evenly strided).

    ⚠ IT TAKES THE TRUE BASE POSE, AND THAT LEAKS NOTHING. The result is the
    scan relative to the base, levelled — which is what the sensor, the IMU's
    gravity vector and the joint encoders give a real robot without knowing
    where it is. Subtracting the true (x, y, yaw) is how the simulator's world
    endpoints are turned back into that relative measurement; the matcher never
    sees the pose itself.
    """
    px = List[Float64]()
    py = List[Float64]()
    var keep = List[Int]()
    for b in range(len(hits)):
        if not hits[b]:
            continue
        var z = endpoints[3 * b + 2]
        if z < z_lo or z > z_hi:
            continue
        keep.append(b)
    var step = max(1, (len(keep) + max_points - 1) // max(1, max_points))
    var c = cos(base_yaw)
    var s = sin(base_yaw)
    for k in range(0, len(keep), step):
        var b = keep[k]
        var dx = endpoints[3 * b] - base_x
        var dy = endpoints[3 * b + 1] - base_y
        px.append(c * dx + s * dy)
        py.append(-s * dx + c * dy)


def retarget_scan(
    mut endpoints: List[Float64],
    mut origin_x: Float64, mut origin_y: Float64,
    from_x: Float64, from_y: Float64, from_yaw: Float64,
    to_x: Float64, to_y: Float64, to_yaw: Float64,
):
    """Re-express a scan's world points (and its origin) as if the base had
    been at `to` instead of `from`: the 2D rigid transform T_to * T_from^-1,
    heights unchanged. SLAM integrates a scan at the ESTIMATED pose this way.

    ⚠ Same argument as `base_frame_points`: the simulator gives world points
    from the TRUE pose; removing that pose and applying the estimate is the
    same as placing the sensor's own relative measurement at the estimate.
    """
    var d = to_yaw - from_yaw
    var c = cos(d)
    var s = sin(d)
    var n = len(endpoints) // 3
    for b in range(n):
        var dx = endpoints[3 * b] - from_x
        var dy = endpoints[3 * b + 1] - from_y
        endpoints[3 * b] = to_x + c * dx - s * dy
        endpoints[3 * b + 1] = to_y + s * dx + c * dy
    var ox = origin_x - from_x
    var oy = origin_y - from_y
    origin_x = to_x + c * ox - s * oy
    origin_y = to_y + s * ox + c * oy
