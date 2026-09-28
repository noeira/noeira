"""The SAP narrow phase's oriented-box reject never drops a touching pair.

    pixi run mojo run -I . tests/physics3d/test_obb_separated.mojo

`broadphase_sap._obb_separated` is MuJoCo's midphase `mj_collideOBB`: a
15-axis separating-axis test run before GJK. Rejecting a pair that touches
would drop a contact silently, so the property pinned first is one-sided —
whenever a point of box A lies inside box B, the test must say "not apart" —
checked on random poses by sampling A. The second half checks it does reject:
two unit cubes, one turned 45 degrees about z, touch at a centre distance of
0.5 + sqrt(2)/2 along x, and an edge resting on a face touches where the
geometry says it does.
"""

from std.math import sqrt, cos, sin, abs
from std.random import random_float64, seed
from std.testing import assert_true, assert_false, TestSuite

from noeira.physics3d.collision.broadphase_sap import _obb_separated

comptime D = DType.float32


def _quat_z(angle: Float64) -> Tuple[Float64, Float64, Float64, Float64]:
    """(x, y, z, w) for a rotation of `angle` about z."""
    return (0.0, 0.0, sin(angle / 2), cos(angle / 2))


def _apart(
    p: Tuple[Float64, Float64, Float64],
    q: Tuple[Float64, Float64, Float64, Float64],
    a: Tuple[Float64, Float64, Float64],
    pj: Tuple[Float64, Float64, Float64],
    qj: Tuple[Float64, Float64, Float64, Float64],
    b: Tuple[Float64, Float64, Float64],
) -> Bool:
    return _obb_separated[D](
        Scalar[D](p[0]), Scalar[D](p[1]), Scalar[D](p[2]),
        Scalar[D](q[0]), Scalar[D](q[1]), Scalar[D](q[2]), Scalar[D](q[3]),
        Scalar[D](a[0]), Scalar[D](a[1]), Scalar[D](a[2]),
        Scalar[D](pj[0]), Scalar[D](pj[1]), Scalar[D](pj[2]),
        Scalar[D](qj[0]), Scalar[D](qj[1]), Scalar[D](qj[2]), Scalar[D](qj[3]),
        Scalar[D](b[0]), Scalar[D](b[1]), Scalar[D](b[2]),
    )


def _rot(
    q: Tuple[Float64, Float64, Float64, Float64],
    v: Tuple[Float64, Float64, Float64],
) -> Tuple[Float64, Float64, Float64]:
    var x = q[0]
    var y = q[1]
    var z = q[2]
    var w = q[3]
    var tx = 2 * (y * v[2] - z * v[1])
    var ty = 2 * (z * v[0] - x * v[2])
    var tz = 2 * (x * v[1] - y * v[0])
    return (
        v[0] + w * tx + (y * tz - z * ty),
        v[1] + w * ty + (z * tx - x * tz),
        v[2] + w * tz + (x * ty - y * tx),
    )


def _rand_quat() -> Tuple[Float64, Float64, Float64, Float64]:
    var x = random_float64(-1, 1)
    var y = random_float64(-1, 1)
    var z = random_float64(-1, 1)
    var w = random_float64(-1, 1)
    var n = sqrt(x * x + y * y + z * z + w * w)
    return (x / n, y / n, z / n, w / n)


def test_never_rejects_an_overlapping_pair() raises:
    seed(7)
    var overlapping = 0
    for _trial in range(3000):
        var p = (random_float64(-0.1, 0.1), random_float64(-0.1, 0.1), random_float64(-0.1, 0.1))
        var pj = (random_float64(-0.1, 0.1), random_float64(-0.1, 0.1), random_float64(-0.1, 0.1))
        var q = _rand_quat()
        var qj = _rand_quat()
        var a = (random_float64(0.005, 0.08), random_float64(0.005, 0.08), random_float64(0.005, 0.08))
        var b = (random_float64(0.005, 0.08), random_float64(0.005, 0.08), random_float64(0.005, 0.08))
        # Is some sampled point of A inside B?
        var inside = False
        var qjc = (-qj[0], -qj[1], -qj[2], qj[3])
        for ix in range(5):
            for iy in range(5):
                for iz in range(5):
                    var lx = a[0] * (Float64(ix) / 2.0 - 1.0)
                    var ly = a[1] * (Float64(iy) / 2.0 - 1.0)
                    var lz = a[2] * (Float64(iz) / 2.0 - 1.0)
                    var w = _rot(q, (lx, ly, lz))
                    var r = _rot(qjc, (w[0] + p[0] - pj[0], w[1] + p[1] - pj[1], w[2] + p[2] - pj[2]))
                    if abs(r[0]) <= b[0] and abs(r[1]) <= b[1] and abs(r[2]) <= b[2]:
                        inside = True
        if inside:
            overlapping += 1
            assert_false(
                _apart(p, q, a, pj, qj, b),
                "an overlapping pair was reported apart — a dropped contact",
            )
    assert_true(overlapping > 300, "fixture: too few overlapping samples to mean anything")


def test_rejects_where_the_geometry_says() raises:
    # Two unit cubes (half-size 0.5), B turned 45 deg about z, centres on x:
    # B reaches sqrt(2)/2 along x, so they touch at 0.5 + 0.70711 = 1.20711.
    var ident = (0.0, 0.0, 0.0, 1.0)
    var q45 = _quat_z(3.141592653589793 / 4)
    var h = (0.5, 0.5, 0.5)
    assert_false(_apart((0.0, 0.0, 0.0), ident, h, (1.20, 0.0, 0.0), q45, h),
                 "overlapping by 7 mm must not be apart")
    assert_true(_apart((0.0, 0.0, 0.0), ident, h, (1.22, 0.0, 0.0), q45, h),
                "13 mm apart must be apart")
    # The inflated side: 13 mm apart with a 20 mm margin folded into A.
    var hm = (0.52, 0.52, 0.52)
    assert_false(_apart((0.0, 0.0, 0.0), ident, hm, (1.22, 0.0, 0.0), q45, h),
                 "a gap inside the margin must not be apart")
    # Edge on face: A turned 45 deg about x puts an EDGE on top at
    # 0.5*sqrt(2) = 0.70711; B turned about z keeps a face at -0.5 below, so
    # they touch at a centre height of 1.20711.
    var qx45 = (sin(3.141592653589793 / 8), 0.0, 0.0, cos(3.141592653589793 / 8))
    assert_false(_apart((0.0, 0.0, 0.0), qx45, h, (0.0, 0.0, 1.20), q45, h),
                 "edge-on-face overlap must not be apart")
    assert_true(_apart((0.0, 0.0, 0.0), qx45, h, (0.0, 0.0, 1.22), q45, h),
                "edge-on-face gap must be apart")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
