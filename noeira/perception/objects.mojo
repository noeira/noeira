"""An object of KNOWN size on the floor, from the depth returns labelled as it:
its place and heading (P3, objects).

`places.mojo` answers "where is the fridge" for a fixture: cells that
accumulate over a tour, a box whose yaw is only defined modulo a quarter turn,
an approach point a metre off a face. A thing the robot is to PICK UP needs
more from less: one image, a place to a few centimetres, and a heading, since
the hands go to two particular faces. What makes that possible is that the
object's size is known.

    fit_box_on_floor(points, hx, hy, view_x, view_y) -> BoxOnFloor

`points` are WORLD points (3 floats each) of the returns labelled as the
object, whatever labelled them (the simulator's geom ids, a detector's mask):
`depth_scan_cpu` + the label + the sensor's world pose (`lidar_mount_world`).
`hx`, `hy` are the box's half sizes along its own x and y. The box stands on
the floor on a known face, so seen from above every return, on its top or on
a side, falls inside its footprint: the fit is the planar one.

THE FIT. The MINIMUM-AREA rectangle around the points, as `oriented_box`
(`places.mojo`): for a heading θ the points have an extent along (cos θ, sin θ)
and one across, and the heading kept is the one whose rectangle is smallest,
searched over a quarter turn every degree and then every tenth around the
best. The KNOWN SIDES then say which of its two axes is the box's x (that is
what a fixture's box cannot say): the assignment whose extents are nearer
(2·hx, 2·hy). The extents are taken between the 2nd and 98th percentiles (a
few returns smeared along their rays by the depth error must not set the
size). The centre is the middle of the extents; where an extent falls SHORT of
the known side (a face cut by the image's edge, or hidden), the missing part
is taken to be on the far side from the viewer and the centre is moved that
way by half of it.

⚠ NOT "the heading whose extents are nearest the known sides": that was the
first version, and it turns the box. Trimmed extents are a little short at the
true heading, and a rotation of 2° stretches them back to the known sides: the
clean test read 2.4° and 1 cm off, and a box seen two thirds deep was fitted
on its diagonal.
⚠ The percentiles are read off a 512-bin histogram, not a sort: the fit is
called at every heading, and the library's sort made one fit take seconds.

DEPTH NOISE: `refine_box_on_floor`. A depth camera's error is along the ray,
and it grows with the square of the distance on a stereo device (1.6 cm at
1.5 m, 6.5 cm at 3 m for the D435 class at 424 x 240). Fitted as they come,
such returns read the box TOWARD the camera by about one standard deviation:
the near edge carries the returns of the near side as well as the top's, so
the smear moves the near percentile out more than the far one (measured in the
room: 1.0, 2.0, 2.8, 6.1 and 13 cm at 1, 1.5, 2, 3 and 4 m, every one toward
the camera; 0.2 mm with a noise-free sensor). The refinement is the model fit
that noise calls for: the box's three faces in view are known planes, and
every return says how far along ITS ray the box was met. For a pose, each
return's ray is cut with the box (its own frame, the slab test); the residual
is the measured length minus that one, held within ±`cap`, and `cap` for a ray
the box misses. The pose that makes the sum of squares least is searched from
the planar fit by steps in x, y and heading that halve from 2 cm and 2° to a
third of a millimetre. Thousands of returns share three numbers, so a
centimetre of noise a return leaves millimetres.

⚠ NOT "put the top's returns on their own plane, then fit" (`flatten_top`,
tried first): which returns are the top's is not known, and the returns of the
near side just under the top edge, moved along their rays to the top's height,
land up to (band / tan elevation) in FRONT of the box: 12 cm for a 5 cm band
seen from 1.25 m at 2 m. It read the box 2.8 cm toward the viewer where the
plain fit read 1.7.

TRACKING: `track_box`. Once the box is picked up it is no longer on the floor
nor level, and the arms hide parts of it. `track_box` is the same model fit
with the box FREE: six numbers, its centre and its attitude, searched from the
pose of the image before (steps in the three translations and the three small
turns about the world's axes, halving from 2 cm and 2°). Two kinds of returns
enter: those labelled as the box (the measured length against the ray's entry
into the box, capped, as above), and those labelled as something ELSE seen
near it (`others`): a ray that reached the floor or a wall THROUGH where the
box is put says the box is not there, and costs `cap`. Without the second
kind a box whose edges are hidden by the arms can slide under the returns that
are left; with it the free space around the box holds it.

THE START MATTERS MORE THAN THE FIT. The residual is capped, so a start more
than a few centimetres off sees a flat cost and the search goes nowhere, or to
a neighbouring wrong pose (a box hung from the face in view). Two helps: the
search runs a first pass with the cap three times wider and steps twice as
long, then the fine one; and `carry_box` moves the last pose rigidly with a
body that holds the box (a hand whose motion the robot knows from its joints),
which is a far better start for a box in the hands than its last pose or its
last velocity (in the room, with the last velocity as the start, the tracked
crate jumped by up to 20 cm several times a carry).

⚠ THE HEADING IS MODULO A HALF TURN: a box has no front. `yaw` is returned in
[0, π); `yaw_nearest` picks the representative nearest a wanted heading.
⚠ A SQUARE footprint (hx = hy) has a heading modulo a QUARTER turn only; the
fit returns one of the two.
⚠ Nothing here is the label. A wrong mask gives a confident wrong box: `n`,
the two extents and `cost` are returned so a caller can refuse a fit whose
extents are far from the known sides.
"""

from std.math import abs, cos, sin, sqrt, pi


@fieldwise_init
struct BoxOnFloor(Copyable, ImplicitlyCopyable, Movable, Writable):
    """A known box's planar pose, fitted to its labelled returns."""

    var found: Bool
    var x: Float64
    var y: Float64
    var yaw: Float64
    """The direction of the box's own x axis, in [0, π)."""
    var n: Int
    """Returns the fit used (after the height cut and the thinning)."""
    var ex: Float64
    """The extent seen along the box's x, metres (the known side is 2·hx)."""
    var ey: Float64
    var cost: Float64
    """(ex − 2·hx)² + (ey − 2·hy)² at the heading kept, m²."""


def yaw_nearest(yaw: Float64, want: Float64) -> Float64:
    """`yaw` or `yaw + π`, whichever is nearer `want` (both name the same box),
    in (want − π/2, want + π/2]."""
    var d = yaw - want
    while d > 0.5 * pi:
        d -= pi
    while d <= -0.5 * pi:
        d += pi
    return want + d


comptime _BINS = 512


def _trimmed(mut w: List[Float64], n: Int, mut hist: List[Int]) -> Tuple[Float64, Float64]:
    """The 2nd and 98th percentiles of w[0..n), to a 512th of its range."""
    var lo = w[0]
    var hi = w[0]
    for i in range(1, n):
        lo = min(lo, w[i])
        hi = max(hi, w[i])
    var span = hi - lo
    if span <= 0.0:
        return (lo, hi)
    for k in range(_BINS):
        hist[k] = 0
    var scale = Float64(_BINS) / span
    for i in range(n):
        hist[min(Int((w[i] - lo) * scale), _BINS - 1)] += 1
    var cut = Int(0.02 * Float64(n) + 0.5)
    var acc = 0
    var a = 0
    while a < _BINS - 1 and acc + hist[a] <= cut:
        acc += hist[a]
        a += 1
    acc = 0
    var b = _BINS - 1
    while b > 0 and acc + hist[b] <= cut:
        acc += hist[b]
        b -= 1
    return (lo + Float64(a) / scale, lo + Float64(b + 1) / scale)


def _extents(
    xs: List[Float64], ys: List[Float64], c: Float64, s: Float64, mut u: List[Float64], mut v: List[Float64], mut hist: List[Int]
) -> List[Float64]:
    """[u_lo, u_hi, v_lo, v_hi] of the points along (c, s) and (−s, c), between the 2nd and 98th percentiles."""
    var n = len(xs)
    for i in range(n):
        u[i] = c * xs[i] + s * ys[i]
        v[i] = -s * xs[i] + c * ys[i]
    var eu = _trimmed(u, n, hist)
    var ev = _trimmed(v, n, hist)
    return [eu[0], eu[1], ev[0], ev[1]]


def fit_box_on_floor(
    points: List[Float64], hx: Float64, hy: Float64, view_x: Float64, view_y: Float64,
    z_min: Float64 = 0.03, min_points: Int = 30, max_points: Int = 1500,
) -> BoxOnFloor:
    """The planar pose of a box of half sizes (`hx`, `hy`) standing on the floor,
    from the world points labelled as it (3 floats each), seen from (`view_x`,
    `view_y`). Points under `z_min` are dropped (the floor at the box's foot);
    more than `max_points` are thinned evenly. `found` is False under `min_points`."""
    var none = BoxOnFloor(False, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0)
    var total = len(points) // 3
    var kept = 0
    for i in range(total):
        if points[3 * i + 2] >= z_min:
            kept += 1
    if kept < min_points:
        return none
    var stride = max(1, (kept + max_points - 1) // max_points)
    var xs = List[Float64]()
    var ys = List[Float64]()
    var k = 0
    for i in range(total):
        if points[3 * i + 2] < z_min:
            continue
        if k % stride == 0:
            xs.append(points[3 * i])
            ys.append(points[3 * i + 1])
        k += 1
    var n = len(xs)
    if n < min_points:
        return none
    var u = List[Float64](length=n, fill=0.0)
    var v = List[Float64](length=n, fill=0.0)

    var hist = List[Int](length=_BINS, fill=0)

    # the smallest rectangle: every degree of a quarter turn, then every tenth around the best
    var best = 1e30
    var best_th = 0.0
    for d in range(90):
        var th = Float64(d) * pi / 180.0
        var e = _extents(xs, ys, cos(th), sin(th), u, v, hist)
        var j = (e[1] - e[0]) * (e[3] - e[2])
        if j < best:
            best = j
            best_th = th
    var coarse = best_th
    for d in range(-15, 16):
        var th = coarse + Float64(d) * 0.1 * pi / 180.0
        var e = _extents(xs, ys, cos(th), sin(th), u, v, hist)
        var j = (e[1] - e[0]) * (e[3] - e[2])
        if j < best:
            best = j
            best_th = th
    # which of the rectangle's axes is the box's x: the known sides say
    var e0 = _extents(xs, ys, cos(best_th), sin(best_th), u, v, hist)
    var as_is = (e0[1] - e0[0] - 2.0 * hx) ** 2 + (e0[3] - e0[2] - 2.0 * hy) ** 2
    var turned = (e0[3] - e0[2] - 2.0 * hx) ** 2 + (e0[1] - e0[0] - 2.0 * hy) ** 2
    if turned < as_is:
        best_th += 0.5 * pi
    var c = cos(best_th)
    var s = sin(best_th)
    var e = _extents(xs, ys, c, s, u, v, hist)
    var cu = 0.5 * (e[0] + e[1])
    var cv = 0.5 * (e[2] + e[3])
    var ex = e[1] - e[0]
    var ey = e[3] - e[2]

    # a side seen short: the missing part is on the far side from the viewer
    var vu = c * view_x + s * view_y
    var vv = -s * view_x + c * view_y
    # Half a centimetre, not two: the trim itself shortens the FAR edge (the near edges carry the side faces' returns as
    # well as the top's, so the 2 % cut falls on the far ones): a clean box read 5 mm toward the viewer until a shortfall
    # of 1 cm was completed too.
    comptime TOL = 0.005
    if ex < 2.0 * hx - TOL:
        cu += (0.5 * (2.0 * hx - ex)) * (1.0 if cu >= vu else -1.0)
    if ey < 2.0 * hy - TOL:
        cv += (0.5 * (2.0 * hy - ey)) * (1.0 if cv >= vv else -1.0)
    var yaw = best_th
    while yaw >= pi:
        yaw -= pi
    while yaw < 0.0:
        yaw += pi
    return BoxOnFloor(True, c * cu - s * cv, s * cu + c * cv, yaw, n, ex, ey, (ex - 2.0 * hx) ** 2 + (ey - 2.0 * hy) ** 2)


def _ray_box(
    ox: Float64, oy: Float64, oz: Float64, dx: Float64, dy: Float64, dz: Float64, hx: Float64, hy: Float64, top: Float64
) -> Float64:
    """The length along a ray (origin and UNIT direction in the box's frame: its centre on the floor at the origin, its
    axes the frame's) at which it enters the box [−hx, hx] x [−hy, hy] x [0, top]; negative = it misses."""
    var t0 = -1e30
    var t1 = 1e30
    if abs(dx) < 1e-12:
        if ox < -hx or ox > hx:
            return -1.0
    else:
        var a = (-hx - ox) / dx
        var b = (hx - ox) / dx
        t0 = max(t0, min(a, b))
        t1 = min(t1, max(a, b))
    if abs(dy) < 1e-12:
        if oy < -hy or oy > hy:
            return -1.0
    else:
        var a = (-hy - oy) / dy
        var b = (hy - oy) / dy
        t0 = max(t0, min(a, b))
        t1 = min(t1, max(a, b))
    if abs(dz) < 1e-12:
        if oz < 0.0 or oz > top:
            return -1.0
    else:
        var a = (0.0 - oz) / dz
        var b = (top - oz) / dz
        t0 = max(t0, min(a, b))
        t1 = min(t1, max(a, b))
    if t1 < t0 or t0 <= 0.0:
        return -1.0
    return t0


def _cost(
    x: Float64, y: Float64, yaw: Float64, dirs: List[Float64], lens: List[Float64], vx: Float64, vy: Float64, vz: Float64,
    hx: Float64, hy: Float64, top: Float64, cap: Float64,
) -> Float64:
    var c = cos(yaw)
    var s = sin(yaw)
    var ox = c * (vx - x) + s * (vy - y)
    var oy = -s * (vx - x) + c * (vy - y)
    var sum = 0.0
    for i in range(len(lens)):
        var dx = c * dirs[3 * i] + s * dirs[3 * i + 1]
        var dy = -s * dirs[3 * i] + c * dirs[3 * i + 1]
        var t = _ray_box(ox, oy, vz, dx, dy, dirs[3 * i + 2], hx, hy, top)
        var r = cap
        if t > 0.0:
            r = min(max(lens[i] - t, -cap), cap)
        sum += r * r
    return sum


def refine_box_on_floor(
    start: BoxOnFloor, points: List[Float64], hx: Float64, hy: Float64, height: Float64,
    view_x: Float64, view_y: Float64, view_z: Float64, cap: Float64 = 0.05, max_points: Int = 1500,
) -> BoxOnFloor:
    """`start` (a `fit_box_on_floor`) refined against the returns themselves: the pose of the box of half sizes (`hx`,
    `hy`) and `height` on the floor whose faces the returns' rays, from the sensor at (`view_x`, `view_y`, `view_z`), meet
    nearest the measured lengths (see the module docstring). `cap`: the most one return may weigh, metres (about three
    standard deviations of the sensor's depth error at that distance). `cost` of the result is the mean squared residual."""
    if not start.found:
        return start
    var total = len(points) // 3
    var stride = max(1, (total + max_points - 1) // max_points)
    var dirs = List[Float64]()
    var lens = List[Float64]()
    for i in range(0, total, stride):
        var dx = points[3 * i] - view_x
        var dy = points[3 * i + 1] - view_y
        var dz = points[3 * i + 2] - view_z
        var l = sqrt(dx * dx + dy * dy + dz * dz)
        if l < 1e-6:
            continue
        dirs.append(dx / l)
        dirs.append(dy / l)
        dirs.append(dz / l)
        lens.append(l)
    if len(lens) < 10:
        return start
    var x = start.x
    var y = start.y
    var yaw = start.yaw
    var best = _cost(x, y, yaw, dirs, lens, view_x, view_y, view_z, hx, hy, height, cap)
    var step = 0.02
    var turn = 2.0 * pi / 180.0
    for _ in range(7):
        for sweep in range(20):
            var moved = False
            for k in range(6):
                var nx = x + (step if k == 0 else (-step if k == 1 else 0.0))
                var ny = y + (step if k == 2 else (-step if k == 3 else 0.0))
                var nw = yaw + (turn if k == 4 else (-turn if k == 5 else 0.0))
                var j = _cost(nx, ny, nw, dirs, lens, view_x, view_y, view_z, hx, hy, height, cap)
                if j < best:
                    best = j
                    x = nx
                    y = ny
                    yaw = nw
                    moved = True
            if not moved:
                break
        step *= 0.5
        turn *= 0.5
    while yaw >= pi:
        yaw -= pi
    while yaw < 0.0:
        yaw += pi
    return BoxOnFloor(True, x, y, yaw, len(lens), start.ex, start.ey, best / Float64(len(lens)))


@fieldwise_init
struct BoxPose(Copyable, ImplicitlyCopyable, Movable, Writable):
    """A free box: its centre and its attitude (unit quaternion, w first), fitted to its returns."""

    var x: Float64
    var y: Float64
    var z: Float64
    var qw: Float64
    var qx: Float64
    var qy: Float64
    var qz: Float64
    var n: Int
    """Returns labelled as the box that the fit used; 0 = none (the pose is the start's)."""
    var cost: Float64
    """Mean squared residual over the returns used (the box's and the others'), m²."""


def _rot_inv(qw: Float64, qx: Float64, qy: Float64, qz: Float64, vx: Float64, vy: Float64, vz: Float64) -> Tuple[Float64, Float64, Float64]:
    """R(q)ᵀ v: a world vector in the frame of the unit quaternion q."""
    # rotate by the conjugate: v' = v + 2 w' (u x v) + 2 u x (u x v), u = −(qx, qy, qz), w' = qw
    var ux = -qx
    var uy = -qy
    var uz = -qz
    var cx = uy * vz - uz * vy
    var cy = uz * vx - ux * vz
    var cz = ux * vy - uy * vx
    var dx = uy * cz - uz * cy
    var dy = uz * cx - ux * cz
    var dz = ux * cy - uy * cx
    return (vx + 2.0 * (qw * cx + dx), vy + 2.0 * (qw * cy + dy), vz + 2.0 * (qw * cz + dz))


def _cost6(
    p: BoxPose, dirs: List[Float64], lens: List[Float64], n_box: Int, vx: Float64, vy: Float64, vz: Float64,
    hx: Float64, hy: Float64, hz: Float64, cap: Float64, margin: Float64,
) -> Float64:
    """The first `n_box` rays are the box's returns, the rest the others'."""
    var o = _rot_inv(p.qw, p.qx, p.qy, p.qz, vx - p.x, vy - p.y, vz - p.z)
    var sum = 0.0
    for i in range(len(lens)):
        var d = _rot_inv(p.qw, p.qx, p.qy, p.qz, dirs[3 * i], dirs[3 * i + 1], dirs[3 * i + 2])
        # the box of the slab test stands on z = 0: shift the origin up by hz
        var t = _ray_box(o[0], o[1], o[2] + hz, d[0], d[1], d[2], hx, hy, 2.0 * hz)
        if i < n_box:
            var r = cap
            if t > 0.0:
                r = min(max(lens[i] - t, -cap), cap)
            sum += r * r
        elif t > 0.0 and t < lens[i] - margin:
            sum += cap * cap
    return sum


def _turned(p: BoxPose, axis: Int, a: Float64) -> BoxPose:
    """`p` turned by the angle `a` about the world's x (0), y (1) or z (2) through its centre."""
    var c = cos(0.5 * a)
    var s = sin(0.5 * a)
    var rx = s if axis == 0 else 0.0
    var ry = s if axis == 1 else 0.0
    var rz = s if axis == 2 else 0.0
    # r ⊗ q
    var w = c * p.qw - rx * p.qx - ry * p.qy - rz * p.qz
    var x = c * p.qx + rx * p.qw + ry * p.qz - rz * p.qy
    var y = c * p.qy - rx * p.qz + ry * p.qw + rz * p.qx
    var z = c * p.qz + rx * p.qy - ry * p.qx + rz * p.qw
    var nrm = sqrt(w * w + x * x + y * y + z * z)
    return BoxPose(p.x, p.y, p.z, w / nrm, x / nrm, y / nrm, z / nrm, p.n, p.cost)


def carry_box(
    p: BoxPose, ox: Float64, oy: Float64, oz: Float64, oqw: Float64, oqx: Float64, oqy: Float64, oqz: Float64,
    nx: Float64, ny: Float64, nz: Float64, nqw: Float64, nqx: Float64, nqy: Float64, nqz: Float64,
) -> BoxPose:
    """`p` moved rigidly with a body that went from the pose (o…) to the pose (n…) (positions and unit quaternions, w
    first, in the world): where a box held by that body is now."""
    # the body's turn in the world: d = n ⊗ conj(o)
    var dw = nqw * oqw + nqx * oqx + nqy * oqy + nqz * oqz
    var dx = -nqw * oqx + nqx * oqw - nqy * oqz + nqz * oqy
    var dy = -nqw * oqy + nqx * oqz + nqy * oqw - nqz * oqx
    var dz = -nqw * oqz - nqx * oqy + nqy * oqx + nqz * oqw
    # R(d) (c − o): by the conjugate's inverse rotation
    var r = _rot_inv(dw, -dx, -dy, -dz, p.x - ox, p.y - oy, p.z - oz)
    var w = dw * p.qw - dx * p.qx - dy * p.qy - dz * p.qz
    var x = dw * p.qx + dx * p.qw + dy * p.qz - dz * p.qy
    var y = dw * p.qy - dx * p.qz + dy * p.qw + dz * p.qx
    var z = dw * p.qz + dx * p.qy - dy * p.qx + dz * p.qw
    var nrm = sqrt(w * w + x * x + y * y + z * z)
    return BoxPose(nx + r[0], ny + r[1], nz + r[2], w / nrm, x / nrm, y / nrm, z / nrm, p.n, p.cost)


def track_box(
    start: BoxPose, points: List[Float64], others: List[Float64], hx: Float64, hy: Float64, hz: Float64,
    view_x: Float64, view_y: Float64, view_z: Float64, cap: Float64 = 0.05, max_points: Int = 600, max_others: Int = 400,
    margin: Float64 = 0.03, min_points: Int = 20,
) -> BoxPose:
    """The free box of half sizes (`hx`, `hy`, `hz`) from the pose `start` (the image before): `points` are the world
    points of the returns labelled as it, `others` those of returns labelled as something else near it (3 floats each),
    both seen from the sensor at (`view_x`, `view_y`, `view_z`). A ray of `others` that passes through the box more than
    `margin` before its own return costs `cap`. With fewer than `min_points` of the box's returns the start is returned
    with `n` = 0: the box is not in view, the caller keeps its own prediction."""
    var total = len(points) // 3
    if total < min_points:
        return BoxPose(start.x, start.y, start.z, start.qw, start.qx, start.qy, start.qz, 0, 0.0)
    var dirs = List[Float64]()
    var lens = List[Float64]()
    var stride = max(1, (total + max_points - 1) // max_points)
    for i in range(0, total, stride):
        var dx = points[3 * i] - view_x
        var dy = points[3 * i + 1] - view_y
        var dz = points[3 * i + 2] - view_z
        var l = sqrt(dx * dx + dy * dy + dz * dz)
        if l < 1e-6:
            continue
        dirs.append(dx / l)
        dirs.append(dy / l)
        dirs.append(dz / l)
        lens.append(l)
    var n_box = len(lens)
    var n_oth = len(others) // 3
    var so = max(1, (n_oth + max_others - 1) // max(1, max_others))
    for i in range(0, n_oth, so):
        var dx = others[3 * i] - view_x
        var dy = others[3 * i + 1] - view_y
        var dz = others[3 * i + 2] - view_z
        var l = sqrt(dx * dx + dy * dy + dz * dz)
        if l < 1e-6:
            continue
        dirs.append(dx / l)
        dirs.append(dy / l)
        dirs.append(dz / l)
        lens.append(l)
    var p = BoxPose(start.x, start.y, start.z, start.qw, start.qx, start.qy, start.qz, n_box, 0.0)
    var best = 0.0
    # a wide pass (the cap three times wider, steps from 4 cm and 4 deg), then the fine one
    for ps in range(2):
        var cp = 3.0 * cap if ps == 0 else cap
        var step = 0.04 if ps == 0 else 0.02
        var turn = (4.0 if ps == 0 else 2.0) * pi / 180.0
        best = _cost6(p, dirs, lens, n_box, view_x, view_y, view_z, hx, hy, hz, cp, margin)
        for _ in range(4 if ps == 0 else 7):
            for sweep in range(12):
                var moved = False
                for k in range(12):
                    var q = p
                    var sg = 1.0 if k % 2 == 0 else -1.0
                    var ax = k // 2
                    if ax == 0:
                        q.x += sg * step
                    elif ax == 1:
                        q.y += sg * step
                    elif ax == 2:
                        q.z += sg * step
                    else:
                        q = _turned(p, ax - 3, sg * turn)
                    var j = _cost6(q, dirs, lens, n_box, view_x, view_y, view_z, hx, hy, hz, cp, margin)
                    if j < best:
                        best = j
                        p = q
                        moved = True
                if not moved:
                    break
            step *= 0.5
            turn *= 0.5
    p.n = n_box
    p.cost = best / Float64(len(lens))
    return p


def box_lowest(p: BoxPose, hx: Float64, hy: Float64, hz: Float64) -> Float64:
    """The height of the lowest corner of the box at `p`. A box tipped up on
    an edge has its CENTRE above its resting height and a corner still on the
    floor: "off the floor" is a statement about this, not about the centre.

    ⚠ Found the hard way: "the centre is 5 cm above its resting height" read a
    crate tipped on an edge by a failed pickup as lifted. Out of view a moment
    later it was carried on with the hand that had left it, and ended metres
    from where it lay, in full view."""
    # the world's up in the box's frame: the half sizes along it
    var u = _rot_inv(p.qw, p.qx, p.qy, p.qz, 0.0, 0.0, 1.0)
    return p.z - (abs(u[0]) * hx + abs(u[1]) * hy + abs(u[2]) * hz)


def returns_off_box(p: BoxPose, points: List[Float64], hx: Float64, hy: Float64, hz: Float64, slack: Float64 = 0.05) -> Bool:
    """True when the centroid of `points` (the returns labelled as the box, 3
    floats each) is further from the centre of `p` than any point of the box
    can be (its half diagonal) plus `slack`: these returns are not on a box at
    `p`. `track_box` is a LOCAL fit whose residuals are capped, so from such a
    start its cost is flat and it returns the start with `n` > 0: the caller
    asks this first, and `find_box` when it is true. False without points."""
    var n = len(points) // 3
    if n == 0:
        return False
    var cx = 0.0
    var cy = 0.0
    var cz = 0.0
    for i in range(n):
        cx += points[3 * i]
        cy += points[3 * i + 1]
        cz += points[3 * i + 2]
    cx /= Float64(n)
    cy /= Float64(n)
    cz /= Float64(n)
    var reach = sqrt(hx * hx + hy * hy + hz * hz) + slack
    return (cx - p.x) ** 2 + (cy - p.y) ** 2 + (cz - p.z) ** 2 > reach * reach


def returns_centre(points: List[Float64], view_x: Float64, view_y: Float64, view_z: Float64, push: Float64) -> Tuple[Float64, Float64, Float64]:
    """Where to start looking for a box from its returns alone: their centroid,
    moved `push` further from the sensor (the returns are on the faces turned
    to it; `push` = the smallest half size). The caller needs it before
    `find_box` to choose the free-space returns around it."""
    var n = len(points) // 3
    if n == 0:
        return (view_x, view_y, view_z)
    var cx = 0.0
    var cy = 0.0
    var cz = 0.0
    for i in range(n):
        cx += points[3 * i]
        cy += points[3 * i + 1]
        cz += points[3 * i + 2]
    cx /= Float64(n)
    cy /= Float64(n)
    cz /= Float64(n)
    var dx = cx - view_x
    var dy = cy - view_y
    var dz = cz - view_z
    var l = sqrt(dx * dx + dy * dy + dz * dz)
    if l < 1e-6:
        return (cx, cy, cz)
    return (cx + push * dx / l, cy + push * dy / l, cz + push * dz / l)


def find_box(
    last: BoxPose, cx: Float64, cy: Float64, cz: Float64, points: List[Float64], others: List[Float64],
    hx: Float64, hy: Float64, hz: Float64, view_x: Float64, view_y: Float64, view_z: Float64, cap: Float64 = 0.05,
) -> BoxPose:
    """The free box from its returns when the last pose is of no use (it was
    lost and is in view again somewhere else: `returns_off_box`). `track_box`
    from 13 starts at (`cx`, `cy`, `cz`) (`returns_centre`), the fit of least
    cost returned: the attitude of `last`, and each of the box's three axes up
    at four headings 45° apart (a box that fell may lie on a side; the local
    fit turns the rest of the way from the nearest). `others` are the free-space returns
    around that centre. `n` = 0 with too few returns, as `track_box`.

    Thirteen fits: for the image where a lost box comes back, not every image."""
    var s = sqrt(0.5)
    # the box's z up; its x up (a quarter about −y); its y up (a quarter about +x)
    var bases: List[BoxPose] = [
        BoxPose(cx, cy, cz, 1.0, 0.0, 0.0, 0.0, 0, 0.0),
        BoxPose(cx, cy, cz, s, 0.0, -s, 0.0, 0, 0.0),
        BoxPose(cx, cy, cz, s, s, 0.0, 0.0, 0, 0.0),
    ]
    var best = track_box(BoxPose(cx, cy, cz, last.qw, last.qx, last.qy, last.qz, 0, 0.0), points, others, hx, hy, hz,
                         view_x, view_y, view_z, cap)
    if best.n == 0:
        return best
    for b in range(3):
        for k in range(4):
            var f = track_box(_turned(bases[b], 2, Float64(k) * 0.25 * pi), points, others, hx, hy, hz, view_x, view_y, view_z, cap)
            if f.n > 0 and f.cost < best.cost:
                best = f
    return best
