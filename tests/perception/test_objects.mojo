"""P3, objects: a box of known size on the floor from its labelled returns.

    pixi run mojo run -I . tests/perception/test_objects.mojo

`fit_box_on_floor` against boxes whose pose is known, the returns made here
(a lattice on the top face and on the two sides that face the viewer: what a
camera above the box sees), no simulator:

* the place within 5 mm and the heading within 0.5° on clean returns, for
  several places, headings and viewers, a crate's sides (0.26 x 0.43 m);
* with every return moved up to 1 cm along its ray from the viewer (a depth
  error), within 1.5 cm and 3°;
* a PARTIAL view, the far third of the box missing: the centre is still
  within 2 cm (the missing part is put on the far side from the viewer);
  the same returns fitted with the viewer placed on the OTHER side move the
  centre the other way (the correction is the viewer's, not a constant);
* THE DEPTH NOISE: with every return moved up to 4 cm along its ray, the plain
  fit reads the box more than 1 cm TOWARD the viewer (the bias the refinement
  is for: asserted, so the test knows the fault it guards); refined against
  the same returns (`refine_box_on_floor`) it is within 5 mm and 1°; and a
  refinement started 3 cm and 4° off the truth on CLEAN returns comes back to
  within 2 mm and 0.3° (it finds the pose, it does not just keep its start);
* TRACKING a free box (`track_box`): a box lifted to 0.9 m and tilted 20° and
  12° about two axes, its returns a lattice on the faces that face the viewer,
  the start 3 cm and 5° off: back within 3 mm and 0.6° on clean returns, and
  within 6 mm and 1.5° with every return moved up to 1 cm along its ray;
  with only a STRIP of its top in view (the rest hidden, as by the arms) the
  fit slides along the strip unless the returns of the floor beyond the box
  are given too: asserted both ways (the others' rays are what holds it);
  too few returns: the start comes back with `n` = 0;
* `carry_box`: a box moved with a body that turns a quarter turn and moves is
  where the rigid motion puts it; a still body leaves it; and `track_box`
  started 12 cm off with a 3 cm cap comes back within 5 mm (the wide pass);
* `yaw_nearest` names the same box by the heading nearest the one asked;
* too few returns: not found.

⚠ The heading is compared modulo a half turn: a box has no front.
"""

from std.math import abs, atan2, cos, sin, sqrt, pi
from std.testing import assert_true, TestSuite

from noeira.perception.objects import (
    BoxOnFloor, BoxPose, carry_box, fit_box_on_floor, refine_box_on_floor, track_box, yaw_nearest, box_lowest, returns_off_box,
    returns_centre, find_box,
)

comptime HX: Float64 = 0.13
comptime HY: Float64 = 0.215
comptime HZ: Float64 = 0.19


def _dyaw(a: Float64, b: Float64) -> Float64:
    """|a − b| modulo a half turn, radians."""
    var d = a - b
    while d > 0.5 * pi:
        d -= pi
    while d <= -0.5 * pi:
        d += pi
    return abs(d)


def _returns(
    x: Float64, y: Float64, yaw: Float64, vx: Float64, vy: Float64, vz: Float64, noise: Float64, cut: Float64
) -> List[Float64]:
    """World points on the box's top and on its sides that face the viewer at
    (vx, vy, vz): a 1 cm lattice. `noise`: each point moved along its ray from
    the viewer by a fixed pseudo-random length in ±noise. `cut`: points whose
    own x coordinate is farther from the viewer's side than this fraction of
    the box's depth are dropped (1 = none)."""
    var out = List[Float64]()
    var c = cos(yaw)
    var s = sin(yaw)
    # the viewer in the box's frame
    var lx = c * (vx - x) + s * (vy - y)
    var ly = -s * (vx - x) + c * (vy - y)
    var loc = List[Float64]()
    var nx = Int(2.0 * HX / 0.01 + 0.5)
    var ny = Int(2.0 * HY / 0.01 + 0.5)
    var nz = Int(2.0 * HZ / 0.01 + 0.5)
    for i in range(nx + 1):
        for j in range(ny + 1):
            loc.append(-HX + Float64(i) * 0.01)
            loc.append(-HY + Float64(j) * 0.01)
            loc.append(2.0 * HZ)
    var sx = HX if lx > 0.0 else -HX
    for j in range(ny + 1):
        for k in range(1, nz):
            loc.append(sx)
            loc.append(-HY + Float64(j) * 0.01)
            loc.append(Float64(k) * 0.01)
    var sy = HY if ly > 0.0 else -HY
    for i in range(nx + 1):
        for k in range(1, nz):
            loc.append(-HX + Float64(i) * 0.01)
            loc.append(sy)
            loc.append(Float64(k) * 0.01)
    var seed: UInt64 = 0x9E3779B97F4A7C15
    for p in range(len(loc) // 3):
        var px = loc[3 * p]
        var py = loc[3 * p + 1]
        var pz = loc[3 * p + 2]
        # the part of the box's depth kept, counted from the viewer's side
        var from_near = (px + HX) / (2.0 * HX) if lx < 0.0 else (HX - px) / (2.0 * HX)
        if from_near > cut:
            continue
        var wx = x + c * px - s * py
        var wy = y + s * px + c * py
        var wz = pz
        if noise > 0.0:
            seed = seed * 6364136223846793005 + 1442695040888963407
            var r = (Float64(Int((seed >> 33) % 2001)) / 1000.0 - 1.0) * noise
            var dx = wx - vx
            var dy = wy - vy
            var dz = wz - vz
            var l = sqrt(dx * dx + dy * dy + dz * dz)
            wx += r * dx / l
            wy += r * dy / l
            wz += r * dz / l
        out.append(wx)
        out.append(wy)
        out.append(wz)
    return out^


def test_clean_returns_give_the_pose() raises:
    var xs: List[Float64] = [0.0, 1.3, -2.1, 0.4]
    var ys: List[Float64] = [0.0, -0.7, 1.9, 2.6]
    var yaws: List[Float64] = [0.0, 0.5, 1.2, 2.9]
    var vxs: List[Float64] = [-1.5, 3.0, -2.0, 0.4]
    var vys: List[Float64] = [0.3, 0.5, 0.0, 0.2]
    for k in range(len(xs)):
        var pts = _returns(xs[k], ys[k], yaws[k], vxs[k], vys[k], 1.25, 0.0, 1.0)
        var b = fit_box_on_floor(pts, HX, HY, vxs[k], vys[k])
        assert_true(b.found, "case " + String(k) + ": not found")
        var e = sqrt((b.x - xs[k]) ** 2 + (b.y - ys[k]) ** 2)
        print("  clean", k, ": place off", e, "m, heading off", _dyaw(b.yaw, yaws[k]) * 180.0 / pi, "deg, extents", b.ex, b.ey, "n", b.n)
        assert_true(e < 0.005, "case " + String(k) + ": place off " + String(e))
        assert_true(_dyaw(b.yaw, yaws[k]) < 0.5 * pi / 180.0, "case " + String(k) + ": heading off " + String(_dyaw(b.yaw, yaws[k])))
        assert_true(b.yaw >= 0.0 and b.yaw < pi, "the heading is not in [0, pi)")


def test_a_centimetre_of_depth_error() raises:
    var xs: List[Float64] = [0.0, 1.3, -2.1]
    var ys: List[Float64] = [0.0, -0.7, 1.9]
    var yaws: List[Float64] = [0.3, 1.9, 2.6]
    var vxs: List[Float64] = [-1.8, 3.0, -2.0]
    var vys: List[Float64] = [0.3, 0.9, 0.0]
    for k in range(len(xs)):
        var pts = _returns(xs[k], ys[k], yaws[k], vxs[k], vys[k], 1.25, 0.01, 1.0)
        var b = fit_box_on_floor(pts, HX, HY, vxs[k], vys[k])
        assert_true(b.found, "case " + String(k) + ": not found")
        var e = sqrt((b.x - xs[k]) ** 2 + (b.y - ys[k]) ** 2)
        print("  1 cm ", k, ": place off", e, "m, heading off", _dyaw(b.yaw, yaws[k]) * 180.0 / pi, "deg")
        assert_true(e < 0.015, "case " + String(k) + ": place off " + String(e))
        assert_true(_dyaw(b.yaw, yaws[k]) < 3.0 * pi / 180.0, "case " + String(k) + ": heading off " + String(_dyaw(b.yaw, yaws[k])))


def test_a_partial_view_is_completed_away_from_the_viewer() raises:
    # the viewer on the box's −x side; the far third of its depth is not seen
    var pts = _returns(0.5, 0.2, 0.0, -1.5, 0.2, 1.25, 0.0, 0.67)
    var b = fit_box_on_floor(pts, HX, HY, -1.5, 0.2)
    assert_true(b.found, "not found")
    var e = sqrt((b.x - 0.5) ** 2 + (b.y - 0.2) ** 2)
    print("  partial: place off", e, "m, extents", b.ex, b.ey)
    assert_true(b.ex < 2.0 * HX - 0.05, "the cut did not shorten the seen extent: " + String(b.ex))
    assert_true(e < 0.02, "place off " + String(e))
    # the control: the same returns with the viewer on the other side are completed the other way
    var o = fit_box_on_floor(pts, HX, HY, 2.5, 0.2)
    print("  partial, viewer on the other side: x", o.x, "against", b.x)
    assert_true(b.x - o.x > 0.05, "the completion does not follow the viewer: " + String(b.x) + " " + String(o.x))


def test_refinement_removes_the_depth_noise_bias() raises:
    var x = 0.4
    var y = -0.3
    var yaw = 0.7
    var vx = -1.6
    var vy = -0.1
    var vz = 1.25
    var pts = _returns(x, y, yaw, vx, vy, vz, 0.04, 1.0)
    var raw = fit_box_on_floor(pts, HX, HY, vx, vy)
    var ln = sqrt((x - vx) ** 2 + (y - vy) ** 2)
    var away_raw = ((raw.x - x) * (x - vx) + (raw.y - y) * (y - vy)) / ln
    var b = refine_box_on_floor(raw, pts, HX, HY, 2.0 * HZ, vx, vy, vz, 0.12)
    var e = sqrt((b.x - x) ** 2 + (b.y - y) ** 2)
    print("  4 cm along the ray: plain fit", away_raw, "m away from the viewer (negative = toward); refined: off", e, "m,",
          _dyaw(b.yaw, yaw) * 180.0 / pi, "deg, mean squared residual", b.cost)
    assert_true(away_raw < -0.01, "the plain fit no longer shows the bias this guards: " + String(away_raw))
    assert_true(e < 0.005, "refined: place off " + String(e))
    assert_true(_dyaw(b.yaw, yaw) < 1.0 * pi / 180.0, "refined: heading off " + String(_dyaw(b.yaw, yaw)))
    # it finds the pose from a wrong start, on clean returns
    var clean = _returns(x, y, yaw, vx, vy, vz, 0.0, 1.0)
    var off = BoxOnFloor(True, x + 0.03, y - 0.01, yaw + 4.0 * pi / 180.0, 0, 0.0, 0.0, 0.0)
    var back = refine_box_on_floor(off, clean, HX, HY, 2.0 * HZ, vx, vy, vz, 0.05)
    var e2 = sqrt((back.x - x) ** 2 + (back.y - y) ** 2)
    print("  from 3 cm and 4 deg off, clean returns: off", e2, "m,", _dyaw(back.yaw, yaw) * 180.0 / pi, "deg")
    assert_true(e2 < 0.002, "from a wrong start: place off " + String(e2))
    assert_true(_dyaw(back.yaw, yaw) < 0.3 * pi / 180.0, "from a wrong start: heading off " + String(_dyaw(back.yaw, yaw)))


def _q_of(rx: Float64, ry: Float64, rz: Float64) -> List[Float64]:
    """The quaternion (w first) of a turn rz about z, then ry about y, then rx about x (each about the WORLD's axis)."""
    var q: List[Float64] = [1.0, 0.0, 0.0, 0.0]
    var ang: List[Float64] = [rz, ry, rx]
    var axs: List[Int] = [2, 1, 0]
    for k in range(3):
        var c = cos(0.5 * ang[k])
        var s = sin(0.5 * ang[k])
        var r: List[Float64] = [c, s if axs[k] == 0 else 0.0, s if axs[k] == 1 else 0.0, s if axs[k] == 2 else 0.0]
        var w = r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3]
        var x = r[0] * q[1] + r[1] * q[0] + r[2] * q[3] - r[3] * q[2]
        var y = r[0] * q[2] - r[1] * q[3] + r[2] * q[0] + r[3] * q[1]
        var z = r[0] * q[3] + r[1] * q[2] - r[2] * q[1] + r[3] * q[0]
        q = [w, x, y, z]
    return q^


def _rot(q: List[Float64], v: List[Float64]) -> List[Float64]:
    """R(q) v by the rotation matrix (not the fitter's formula)."""
    var w = q[0]
    var x = q[1]
    var y = q[2]
    var z = q[3]
    return [
        (1.0 - 2.0 * (y * y + z * z)) * v[0] + 2.0 * (x * y - w * z) * v[1] + 2.0 * (x * z + w * y) * v[2],
        2.0 * (x * y + w * z) * v[0] + (1.0 - 2.0 * (x * x + z * z)) * v[1] + 2.0 * (y * z - w * x) * v[2],
        2.0 * (x * z - w * y) * v[0] + 2.0 * (y * z + w * x) * v[1] + (1.0 - 2.0 * (x * x + y * y)) * v[2],
    ]


def _free_returns(c: List[Float64], q: List[Float64], v: List[Float64], noise: Float64, strip: Float64) -> List[Float64]:
    """World points on the faces of the box (centre c, attitude q) that face the viewer v: a 1 cm lattice on each.
    `strip` > 0: only the TOP face, and of it only the points whose own y is within ±strip (the rest is hidden)."""
    var out = List[Float64]()
    var h: List[Float64] = [HX, HY, HZ]
    var seed: UInt64 = 0x2545F4914F6CDD1D
    for face in range(6):
        var ax = face // 2
        var sg = 1.0 if face % 2 == 0 else -1.0
        if strip > 0.0 and not (ax == 2 and sg > 0.0):
            continue
        var nl: List[Float64] = [0.0, 0.0, 0.0]
        nl[ax] = sg
        var nw = _rot(q, nl)
        var fc_l: List[Float64] = [0.0, 0.0, 0.0]
        fc_l[ax] = sg * h[ax]
        var fc = _rot(q, fc_l)
        # the face is seen when the viewer is on its outer side
        if nw[0] * (v[0] - c[0] - fc[0]) + nw[1] * (v[1] - c[1] - fc[1]) + nw[2] * (v[2] - c[2] - fc[2]) <= 0.0:
            continue
        var a1 = (ax + 1) % 3
        var a2 = (ax + 2) % 3
        var n1 = Int(2.0 * h[a1] / 0.01 + 0.5)
        var n2 = Int(2.0 * h[a2] / 0.01 + 0.5)
        for i in range(n1 + 1):
            for j in range(n2 + 1):
                var pl: List[Float64] = [0.0, 0.0, 0.0]
                pl[ax] = sg * h[ax]
                pl[a1] = -h[a1] + Float64(i) * 0.01
                pl[a2] = -h[a2] + Float64(j) * 0.01
                if strip > 0.0 and abs(pl[1]) > strip:
                    continue
                var pw = _rot(q, pl)
                var wx = c[0] + pw[0]
                var wy = c[1] + pw[1]
                var wz = c[2] + pw[2]
                if noise > 0.0:
                    seed = seed * 6364136223846793005 + 1442695040888963407
                    var r = (Float64(Int((seed >> 33) % 2001)) / 1000.0 - 1.0) * noise
                    var dx = wx - v[0]
                    var dy = wy - v[1]
                    var dz = wz - v[2]
                    var l = sqrt(dx * dx + dy * dy + dz * dz)
                    wx += r * dx / l
                    wy += r * dy / l
                    wz += r * dz / l
                out.append(wx)
                out.append(wy)
                out.append(wz)
    return out^


def _att_err_deg(a: BoxPose, q: List[Float64]) -> Float64:
    """The angle between two attitudes, degrees."""
    var d = abs(a.qw * q[0] + a.qx * q[1] + a.qy * q[2] + a.qz * q[3])
    return 2.0 * atan2(sqrt(max(0.0, 1.0 - d * d)), d) * 180.0 / pi


def test_track_a_free_box() raises:
    var c: List[Float64] = [0.35, 0.05, 0.9]
    var q = _q_of(20.0 * pi / 180.0, 12.0 * pi / 180.0, 0.6)
    var v: List[Float64] = [-0.1, 0.0, 1.45]
    var qs = _q_of(24.0 * pi / 180.0, 9.0 * pi / 180.0, 0.6 + 3.0 * pi / 180.0)
    var start = BoxPose(c[0] + 0.02, c[1] - 0.02, c[2] + 0.01, qs[0], qs[1], qs[2], qs[3], 0, 0.0)
    var none = List[Float64]()
    var noises: List[Float64] = [0.0, 0.01]
    var tol_m: List[Float64] = [0.003, 0.006]
    var tol_deg: List[Float64] = [0.6, 1.5]
    for k in range(2):
        var pts = _free_returns(c, q, v, noises[k], 0.0)
        var b = track_box(start, pts, none, HX, HY, HZ, v[0], v[1], v[2], 0.05)
        var e = sqrt((b.x - c[0]) ** 2 + (b.y - c[1]) ** 2 + (b.z - c[2]) ** 2)
        print("  free box, noise", noises[k], ": centre off", e, "m, attitude off", _att_err_deg(b, q), "deg, from",
              sqrt(0.02 ** 2 + 0.02 ** 2 + 0.01 ** 2), "m and", _att_err_deg(start, q), "deg;", b.n, "returns")
        assert_true(b.n > 0, "the box was not fitted")
        assert_true(e < tol_m[k], "centre off " + String(e))
        assert_true(_att_err_deg(b, q) < tol_deg[k], "attitude off " + String(_att_err_deg(b, q)))
    var few = List[Float64]()
    for i in range(5):
        few.append(c[0])
        few.append(c[1])
        few.append(c[2] + HZ)
    assert_true(track_box(start, few, none, HX, HY, HZ, v[0], v[1], v[2]).n == 0, "five returns must not be a fit")


def test_the_others_rays_hold_a_box_seen_in_part() raises:
    # a level box in the hands' place; only a strip of its top, 6 cm either side of its x axis, is in view
    var c: List[Float64] = [0.35, 0.0, 0.9]
    var q = _q_of(0.0, 0.0, 0.0)
    var v: List[Float64] = [-0.1, 0.0, 1.45]
    var pts = _free_returns(c, q, v, 0.0, 0.06)
    # the start is 6 cm along the box's y: every return of the strip still lies on the shifted box's top
    var start = BoxPose(c[0], c[1] + 0.06, c[2], 1.0, 0.0, 0.0, 0.0, 0, 0.0)
    var none = List[Float64]()
    var alone = track_box(start, pts, none, HX, HY, HZ, v[0], v[1], v[2], 0.05)
    # the floor beyond the box's true sides, as the camera sees it: two rows of returns on z = 0 just outside ±HY
    var others = List[Float64]()
    for i in range(60):
        for side in range(2):
            var px = c[0] - 0.2 + Float64(i) * 0.01
            var py = (HY + 0.03) if side == 0 else -(HY + 0.03)
            # the floor point this ray reaches: the line from the viewer through (px, py, top) continued to z = 0
            var tz = c[2] + HZ
            var t = v[2] / (v[2] - tz)
            others.append(v[0] + t * (px - v[0]))
            others.append(v[1] + t * (py - v[1]))
            others.append(0.0)
    var held = track_box(start, pts, others, HX, HY, HZ, v[0], v[1], v[2], 0.05)
    print("  a strip of the top: alone, y off", alone.y - c[1], "m; with the floor's returns beside the box,", held.y - c[1], "m")
    assert_true(abs(alone.y - c[1]) > 0.03, "the strip alone no longer slides: the control is void (" + String(alone.y - c[1]) + ")")
    assert_true(abs(held.y - c[1]) < 0.035, "the others' rays did not hold the box: " + String(held.y - c[1]))
    assert_true(abs(held.y - c[1]) < abs(alone.y - c[1]) - 0.02, "the others' rays changed nothing")


def test_carry_box_moves_the_box_with_the_body() raises:
    # a body turned a quarter turn about z and moved by (1, 2, 0.5): a box 0.3 m ahead of it (its +x) ends 0.3 m to its left (+y)
    var p = BoxPose(0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 7, 0.0)
    var h = sqrt(0.5)
    var c = carry_box(p, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.5, h, 0.0, 0.0, h)
    print("  carried:", c.x, c.y, c.z, "| attitude", c.qw, c.qx, c.qy, c.qz)
    assert_true(abs(c.x - 1.0) < 1e-12 and abs(c.y - 2.3) < 1e-12 and abs(c.z - 0.5) < 1e-12, "the centre did not turn with the body")
    assert_true(abs(c.qw - h) < 1e-12 and abs(c.qz - h) < 1e-12 and abs(c.qx) < 1e-12 and abs(c.qy) < 1e-12, "the attitude did not turn with the body")
    # a body that did not move leaves the box where it was
    var s2 = carry_box(p, 0.4, -0.2, 1.0, h, h, 0.0, 0.0, 0.4, -0.2, 1.0, h, h, 0.0, 0.0)
    assert_true(abs(s2.x - 0.3) < 1e-12 and abs(s2.y) < 1e-12 and abs(s2.z) < 1e-12 and abs(s2.qw - 1.0) < 1e-12, "a still body moved the box")
    # the fit from a start 12 cm off: the wide pass reaches it (a 3 cm cap alone sees a flat cost there)
    var cc: List[Float64] = [0.35, 0.05, 0.9]
    var q = _q_of(0.1, 0.05, 0.6)
    var v: List[Float64] = [-0.1, 0.0, 1.45]
    var pts = _free_returns(cc, q, v, 0.0, 0.0)
    var none = List[Float64]()
    var far = BoxPose(cc[0] + 0.07, cc[1] + 0.07, cc[2] - 0.07, q[0], q[1], q[2], q[3], 0, 0.0)
    var b = track_box(far, pts, none, HX, HY, HZ, v[0], v[1], v[2], 0.03)
    var e = sqrt((b.x - cc[0]) ** 2 + (b.y - cc[1]) ** 2 + (b.z - cc[2]) ** 2)
    print("  from 12 cm off with a 3 cm cap: centre off", e, "m")
    assert_true(e < 0.005, "the wide pass did not bring a 12 cm start back: " + String(e))


def test_yaw_nearest_and_too_few() raises:
    assert_true(abs(yaw_nearest(0.1, 3.0) - (0.1 + pi)) < 1e-12, "0.1 near 3.0 is 0.1 + pi")
    assert_true(abs(yaw_nearest(3.0, 0.0) - (3.0 - pi)) < 1e-12, "3.0 near 0 is 3.0 - pi")
    assert_true(abs(yaw_nearest(0.4, 0.5) - 0.4) < 1e-12, "0.4 near 0.5 stays")
    var few = List[Float64]()
    for i in range(10):
        few.append(Float64(i) * 0.01)
        few.append(0.0)
        few.append(0.3)
    assert_true(not fit_box_on_floor(few, HX, HY, -1.0, 0.0).found, "ten returns must not be a box")
    # returns on the floor only are not the box either
    var floor = List[Float64]()
    for i in range(200):
        floor.append(Float64(i % 20) * 0.01)
        floor.append(Float64(i // 20) * 0.01)
        floor.append(0.0)
    assert_true(not fit_box_on_floor(floor, HX, HY, -1.0, 0.0).found, "floor returns must not be a box")


def _corner_gap(b: BoxPose, c: List[Float64], q: List[Float64]) -> Float64:
    """The furthest any corner of the true box (c, q) is from the nearest corner of the fitted one: the same for every
    attitude that shows the same box (a box turned half a turn about one of its axes is the same box)."""
    var bq: List[Float64] = [b.qw, b.qx, b.qy, b.qz]
    var worst = 0.0
    for i in range(8):
        var l: List[Float64] = [HX if i % 2 == 0 else -HX, HY if (i // 2) % 2 == 0 else -HY, HZ if i // 4 == 0 else -HZ]
        var t = _rot(q, l)
        var near = 1e9
        for j in range(8):
            var m: List[Float64] = [HX if j % 2 == 0 else -HX, HY if (j // 2) % 2 == 0 else -HY, HZ if j // 4 == 0 else -HZ]
            var f = _rot(bq, m)
            var d = sqrt((c[0] + t[0] - b.x - f[0]) ** 2 + (c[1] + t[1] - b.y - f[1]) ** 2 + (c[2] + t[2] - b.z - f[2]) ** 2)
            near = min(near, d)
        worst = max(worst, near)
    return worst


def test_box_lowest_is_the_corner_not_the_centre() raises:
    # standing on the floor
    assert_true(abs(box_lowest(BoxPose(0.0, 0.0, HZ, 1.0, 0.0, 0.0, 0.0, 0, 0.0), HX, HY, HZ)) < 1e-12, "a standing box touches the floor")
    # tipped 45 degrees about its x, an edge on the floor: the centre is 9.6 cm above its resting height
    var s8 = sin(pi / 8.0)
    var c8 = cos(pi / 8.0)
    var zc = (HY + HZ) * sqrt(0.5)
    var tipped = BoxPose(0.0, 0.0, zc, c8, s8, 0.0, 0.0, 0, 0.0)
    print("  tipped on an edge: centre", zc, "m (resting", HZ, "), lowest corner", box_lowest(tipped, HX, HY, HZ))
    assert_true(zc > HZ + 0.05, "the case is the one the centre's height reads as lifted")
    assert_true(abs(box_lowest(tipped, HX, HY, HZ)) < 1e-12, "a box on its edge touches the floor")
    # in the hands, level and tilted: off the floor either way
    assert_true(abs(box_lowest(BoxPose(0.3, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0, 0, 0.0), HX, HY, HZ) - (0.9 - HZ)) < 1e-12, "a level box in the hands")
    assert_true(box_lowest(BoxPose(0.3, 0.0, 0.9, c8, s8, 0.0, 0.0, 0, 0.0), HX, HY, HZ) > 0.6, "a tilted box in the hands")


def test_a_lost_box_is_found_from_its_returns() raises:
    # the box lies on a side (its y up) 1.3 m ahead of the viewer, turned 0.6 rad; it was last believed level in the hands
    var cz = cos(0.3)
    var sz = sin(0.3)
    var h = sqrt(0.5)
    var q: List[Float64] = [cz * h, cz * h, sz * h, sz * h]      # Rz(0.6) after Rx(90 deg)
    var c: List[Float64] = [1.2, 0.4, HY]
    var v: List[Float64] = [0.0, 0.0, 1.3]
    var pts = _free_returns(c, q, v, 0.005, 0.0)
    var none = List[Float64]()
    var last = BoxPose(-0.3, 0.4, 0.9, 1.0, 0.0, 0.0, 0.0, 0, 0.0)
    var truth = BoxPose(c[0], c[1], c[2], q[0], q[1], q[2], q[3], 0, 0.0)
    assert_true(returns_off_box(last, pts, HX, HY, HZ), "returns 1.5 m from the box must be off it")
    assert_true(not returns_off_box(truth, pts, HX, HY, HZ), "the box's own returns are on it")
    assert_true(not returns_off_box(last, none, HX, HY, HZ), "no returns: nothing to say")
    # the local fit from there: a flat cost, the start comes back as a fit
    var stuck = track_box(last, pts, none, HX, HY, HZ, v[0], v[1], v[2], 0.05)
    var e0 = sqrt((stuck.x - c[0]) ** 2 + (stuck.y - c[1]) ** 2 + (stuck.z - c[2]) ** 2)
    assert_true(stuck.n > 0 and e0 > 1.0, "the case is the one the local fit cannot leave: " + String(e0))
    var s = returns_centre(pts, v[0], v[1], v[2], HX)
    var f = find_box(last, s[0], s[1], s[2], pts, none, HX, HY, HZ, v[0], v[1], v[2], 0.05)
    var e = sqrt((f.x - c[0]) ** 2 + (f.y - c[1]) ** 2 + (f.z - c[2]) ** 2)
    print("  lost 1.5 m away, lying on a side: the local fit stays", e0, "m off; found again: centre off", e, "m, corners off",
          _corner_gap(f, c, q), "m, lowest corner", box_lowest(f, HX, HY, HZ), "m;", f.n, "returns")
    assert_true(f.n > 0, "not found")
    assert_true(e < 0.02, "centre off " + String(e))
    assert_true(_corner_gap(f, c, q) < 0.03, "corners off " + String(_corner_gap(f, c, q)))
    assert_true(abs(box_lowest(f, HX, HY, HZ)) < 0.03, "a box lying on the floor must read as on the floor")
    # too few returns: not a fit
    var few = List[Float64]()
    for i in range(5):
        few.append(c[0])
        few.append(c[1])
        few.append(c[2])
    assert_true(find_box(last, c[0], c[1], c[2], few, none, HX, HY, HZ, v[0], v[1], v[2]).n == 0, "five returns must not be a find")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
