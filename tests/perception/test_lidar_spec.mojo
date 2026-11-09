"""The lidar as data: spec files, scan patterns, the return model.

    pixi run mojo run -I . tests/perception/test_lidar_spec.mojo

`test_lidar_vs_mujoco` takes the sensor-frame directions as GIVEN and gates the
geometry; this file gates the directions themselves (counts, unit length, the
elevations each pattern promises) and `lidar_return` (window, dropout, noise),
which the MuJoCo gate switches off.
"""

from std.math import abs, sqrt, asin, atan2, pi
from std.testing import assert_true, assert_equal, assert_raises, TestSuite

from noeira.perception import (
    LidarSpec,
    LIDAR_FAN,
    LIDAR_RINGS,
    LIDAR_SPIRAL,
    LIDAR_NO_RETURN,
    lidar_spec_from_text,
    load_lidar_spec,
    lidar_directions,
    lidar_return,
)

comptime ZOO = "noeira/perception/zoo/"


def _deg(x: Float64) -> Float64:
    return x * 180.0 / pi


def test_zoo_specs_load_and_scan() raises:
    """Every spec in the zoo parses, validates, and yields unit directions in
    the number it declares."""
    var names: List[String] = [
        "planar_2d_12m.lidar", "rings16_100m.lidar", "spiral_mid360_class.lidar"
    ]
    var expect: List[Int] = [800, 16 * 1800, 20000]
    for k in range(len(names)):
        var s = load_lidar_spec(ZOO + names[k])
        print("  ", s)
        assert_equal(s.rays_per_frame(), expect[k])
        var d = lidar_directions(s, 0)
        assert_equal(len(d), 3 * expect[k])
        var worst = 0.0
        for i in range(expect[k]):
            var n = sqrt(d[3 * i] ** 2 + d[3 * i + 1] ** 2 + d[3 * i + 2] ** 2)
            worst = max(worst, abs(n - 1.0))
        assert_true(worst < 1e-12, names[k] + ": a direction is not unit, " + String(worst))


def test_fan_is_one_plane_all_round() raises:
    var s = load_lidar_spec(ZOO + "planar_2d_12m.lidar")
    var d = lidar_directions(s)
    var n = s.rays_per_frame()
    for i in range(n):
        assert_true(d[3 * i + 2] == 0.0, "a fan beam leaves the plane")
    # All round, no duplicate endpoint: consecutive azimuths 360/n apart, and
    # the last one 360/n short of the first.
    var step = 360.0 / Float64(n)
    for i in range(n):
        var j = (i + 1) % n
        var a0 = _deg(atan2(d[3 * i + 1], d[3 * i]))
        var a1 = _deg(atan2(d[3 * j + 1], d[3 * j]))
        var da = a1 - a0
        if da < 0:
            da += 360.0
        assert_true(abs(da - step) < 1e-9, "azimuth step " + String(da) + " at " + String(i))


def test_partial_fov_spans_its_edges() raises:
    var s = lidar_spec_from_text(
        "pattern = fan\nh_fov_deg = 270\nh_samples = 541\nrange_max = 10\n"
    )
    var d = lidar_directions(s)
    var first = _deg(atan2(d[1], d[0]))
    var last = _deg(atan2(d[3 * 540 + 1], d[3 * 540]))
    assert_true(abs(first + 135.0) < 1e-9 and abs(last - 135.0) < 1e-9,
                "a 270 deg fan spans " + String(first) + " .. " + String(last))


def test_rings_sit_at_their_elevations() raises:
    var s = load_lidar_spec(ZOO + "rings16_100m.lidar")
    var d = lidar_directions(s)
    for r in range(16):
        var want = -15.0 + 2.0 * Float64(r)
        for i in range(0, 1800, 97):
            var k = r * 1800 + i
            var el = _deg(asin(d[3 * k + 2]))
            assert_true(abs(el - want) < 1e-9, "ring " + String(r) + " at " + String(el))


def test_spiral_stays_in_band_and_moves_on() raises:
    """Inside [v_min, v_max]; sin(el) and azimuth roughly uniform; and frame
    1 is not frame 0 (a non-repetitive scanner keeps filling in)."""
    var s = load_lidar_spec(ZOO + "spiral_mid360_class.lidar")
    var d0 = lidar_directions(s, 0)
    var d1 = lidar_directions(s, 1)
    var n = s.rays_per_frame()
    var lo = 1e9
    var hi = -1e9
    var hist = List[Int](length=8, fill=0)
    var same = 0
    for i in range(n):
        var el = _deg(asin(d0[3 * i + 2]))
        lo = min(lo, el)
        hi = max(hi, el)
        var az = atan2(d0[3 * i + 1], d0[3 * i])
        hist[min(7, Int((az + pi) / (2.0 * pi) * 8.0))] += 1
        if abs(d0[3 * i] - d1[3 * i]) < 1e-12 and abs(d0[3 * i + 1] - d1[3 * i + 1]) < 1e-12:
            same += 1
    var hs = String("")
    for k in range(8):
        hs += String(hist[k]) + " "
    print("   spiral elevation range", lo, "..", hi, " azimuth octants", hs)
    assert_true(lo >= -7.0 - 1e-9 and hi <= 52.0 + 1e-9, "spiral leaves its band")
    assert_true(lo < -6.5 and hi > 51.5, "spiral does not reach its band edges")
    for k in range(8):
        assert_true(abs(hist[k] - n // 8) < n // 40, "azimuth octant " + String(k) + " holds " + String(hist[k]))
    assert_true(same == 0, String(same) + " directions repeat between frames 0 and 1")


def test_a_bad_spec_is_refused() raises:
    with assert_raises(contains="unknown key"):
        _ = lidar_spec_from_text("pattern = fan\nrange_sigam = 0.02\n")
    with assert_raises(contains="no `pattern =`"):
        _ = lidar_spec_from_text("h_samples = 10\n")
    with assert_raises(contains="one ring is a fan"):
        _ = lidar_spec_from_text("pattern = rings\nv_samples = 1\nv_min_deg = -1\nv_max_deg = 1\n")
    with assert_raises(contains="range_min < range_max"):
        _ = lidar_spec_from_text("pattern = fan\nrange_min = 5\nrange_max = 2\n")
    with assert_raises(contains="pattern must be"):
        _ = lidar_spec_from_text("pattern = rosette\n")


def test_return_window_dropout_and_noise() raises:
    comptime F = DType.float64
    # Window, no noise: a miss stays a miss, the edges are kept, outside is lost.
    assert_true(lidar_return[F](-1.0, 0.1, 10.0, 0.0, 0.0, 7, 0, 0, 0) == LIDAR_NO_RETURN)
    assert_true(lidar_return[F](0.05, 0.1, 10.0, 0.0, 0.0, 7, 0, 0, 0) == LIDAR_NO_RETURN)
    assert_true(lidar_return[F](10.5, 0.1, 10.0, 0.0, 0.0, 7, 0, 0, 0) == LIDAR_NO_RETURN)
    assert_true(lidar_return[F](3.0, 0.1, 10.0, 0.0, 0.0, 7, 0, 0, 0) == 3.0)

    # Noise: mean ~0 and sd ~sigma over many beams; dropout ~p; deterministic.
    var sigma = 0.02
    var p = 0.1
    var n = 200000
    var sum = 0.0
    var sq = 0.0
    var kept = 0
    for i in range(n):
        var r = lidar_return[F](5.0, 0.0, 100.0, sigma, p, 1234, 3, i % 7, i)
        if r < 0:
            continue
        kept += 1
        sum += r - 5.0
        sq += (r - 5.0) ** 2
    var mean = sum / Float64(kept)
    var sd = sqrt(sq / Float64(kept) - mean * mean)
    var drop = 1.0 - Float64(kept) / Float64(n)
    print("   noise mean", mean, " sd", sd, " dropout", drop)
    assert_true(abs(mean) < 6.0 * sigma / sqrt(Float64(kept)), "noise mean " + String(mean))
    assert_true(abs(sd - sigma) < 0.01 * sigma, "noise sd " + String(sd))
    assert_true(abs(drop - p) < 0.005, "dropout " + String(drop))
    var a = lidar_return[F](5.0, 0.0, 100.0, sigma, p, 1234, 3, 2, 99)
    var b = lidar_return[F](5.0, 0.0, 100.0, sigma, p, 1234, 3, 2, 99)
    var c = lidar_return[F](5.0, 0.0, 100.0, sigma, p, 1235, 3, 2, 99)
    assert_true(a == b, "the same ray of the same frame drew different noise")
    assert_true(a != c, "a different seed drew the same noise")
    # float32 and float64 draw the same uniforms (24-bit draws), so the two
    # dtypes agree to float32 precision.
    var a32 = lidar_return[DType.float32](5.0, 0.0, 100.0, 0.02, 0.1, 1234, 3, 2, 99)
    assert_true(abs(Float64(a32) - a) < 1e-5, "float32 noise " + String(a32) + " vs " + String(a))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
